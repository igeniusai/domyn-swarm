# Copyright 2025 iGenius S.p.A
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import sys

import pytest
from typer.testing import CliRunner

from domyn_swarm.cli.main import app
from domyn_swarm.exceptions import JobNotFoundError

CLI_MODPATH = "domyn_swarm.cli.main"

runner = CliRunner()


class _DummySwarm:
    def __init__(self):
        self.calls = []
        self.deleted_name = None

    def down(self):
        self.calls.append("down")

    def delete_record(self, *, deployment_name: str):
        self.calls.append("delete_record")
        self.deleted_name = deployment_name


class _FakeStateManager:
    # will be set by tests
    _swarm_instance = _DummySwarm()

    @classmethod
    def load(cls, *, deployment_name: str):
        cls._swarm_instance.calls.append(("load", deployment_name))
        return cls._swarm_instance


def test_cli_version(monkeypatch):
    monkeypatch.setattr("domyn_swarm.cli.main.get_version", lambda: "1.2.3")

    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "domyn-swarm CLI Version: " in result.stdout
    assert "1.2.3" in result.stdout


def test_cli_version_short(monkeypatch):
    monkeypatch.setattr("domyn_swarm.cli.main.get_version", lambda: "1.2.3")

    result = runner.invoke(app, ["version", "--short"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "1.2.3"


def test_cli_up_requires_config(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("fake_yaml: true")

    def dummy_load_config(*args, **kwargs):
        class DummyCfg:
            replicas = 1

        return DummyCfg()

    def dummy_start_swarm(cfg, reverse_proxy):
        assert cfg.replicas == 1
        assert reverse_proxy is False

    from domyn_swarm.cli import main

    monkeypatch.setattr(main, "_load_swarm_config", dummy_load_config)
    monkeypatch.setattr(main, "_start_swarm", dummy_start_swarm)

    result = runner.invoke(app, ["up", "-c", str(config)])
    assert result.exit_code == 0


def _reload_cli_with_fake_state_manager(mocker):
    # Ensure fresh import each test so our monkeypatch lands cleanly
    sys.modules.pop(CLI_MODPATH, None)
    cli_module = importlib.import_module(CLI_MODPATH)
    # Patch the symbol as referenced by the CLI module
    mocker.patch.object(cli_module, "SwarmStateManager", _FakeStateManager)
    return cli_module


def test_down_happy_path_invokes_swarm_and_deletes(mocker):
    cli_module = _reload_cli_with_fake_state_manager(mocker)
    app = getattr(cli_module, "app")

    # fresh dummy swarm per test
    _FakeStateManager._swarm_instance = _DummySwarm()
    name = "my-swarm"

    result = runner.invoke(app, ["down", name])

    assert result.exit_code == 0, result.output
    # Success message printed
    assert "✅ Swarm shutdown request sent." in result.output

    swarm = _FakeStateManager._swarm_instance
    # Ensure load was called with the right name
    assert ("load", name) in swarm.calls
    # Ensure down and delete_record were called
    assert "down" in swarm.calls
    assert "delete_record" in swarm.calls
    # Ensure delete_record got the same name
    assert swarm.deleted_name == name


def test_down_bubbles_up_load_errors(mocker):
    cli_module = _reload_cli_with_fake_state_manager(mocker)
    app = getattr(cli_module, "app")

    # Make load raise to simulate missing/invalid state
    def boom(**kwargs):
        raise JobNotFoundError("no such swarm")

    mocker.patch.object(cli_module, "SwarmStateManager", autospec=True)
    cli_module.SwarmStateManager.load.side_effect = boom  # type: ignore[attr-defined]

    with pytest.raises(JobNotFoundError):
        result = runner.invoke(app, ["down", "missing-swarm"], catch_exceptions=False)
        assert result.exit_code != 0


def test_down_order_calls_down_before_delete_record(mocker):
    cli_module = _reload_cli_with_fake_state_manager(mocker)
    app = getattr(cli_module, "app")

    # instrument the dummy to record strict order
    class OrderedSwarm(_DummySwarm):
        def down(self):
            super().down()

        def delete_record(self, *, deployment_name: str):
            super().delete_record(deployment_name=deployment_name)

    _FakeStateManager._swarm_instance = OrderedSwarm()

    result = runner.invoke(app, ["down", "swarm-x"])
    assert result.exit_code == 0, result.output

    calls = _FakeStateManager._swarm_instance.calls
    # First a load tuple, then 'down', then 'delete_record'
    # We only assert relative order of down vs delete_record
    assert "down" in calls
    assert "delete_record" in calls
    assert calls.index("down") < calls.index("delete_record")
