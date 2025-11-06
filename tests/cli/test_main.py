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
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import domyn_swarm.cli.main as main
from domyn_swarm.cli.main import app
from domyn_swarm.exceptions import JobNotFoundError

CLI_MODPATH = "domyn_swarm.cli.main"

runner = CliRunner()


class _DummySwarm:
    def __init__(self, name: str = "dummy-swarm", status_phase: str = "RUNNING"):
        self.calls = []
        self.deleted_name = None
        self.down_called = False
        self.name = name
        self.status_phase = status_phase

    def down(self):
        self.down_called = True
        self.calls.append("down")
        self._delete_record()

    def _delete_record(self):
        self.calls.append("delete_record")
        self.deleted_name = self.name

    def status(self):
        return SimpleNamespace(phase=self.status_phase)


class _DummyCfg:
    def __init__(self, name: str):
        self.name = name


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


def test_cli_up_requires_config(tmp_path, mocker):
    # Arrange
    config = tmp_path / "config.yaml"
    config.write_text("fake_yaml: true")

    class DummyCfg:
        replicas = 1

    from domyn_swarm.cli import main

    # Patch the loader to return a single known instance
    cfg_instance = DummyCfg()
    load_mock = mocker.patch.object(
        main, "_load_swarm_config", return_value=cfg_instance
    )

    # Patch DomynLLMSwarm as a context manager
    fake_swarm = mocker.MagicMock()
    fake_swarm.__enter__.return_value = fake_swarm
    fake_swarm.__exit__.return_value = None
    swarm_cls = mocker.patch.object(main, "DomynLLMSwarm", return_value=fake_swarm)

    runner = CliRunner()

    # Act
    result = runner.invoke(main.app, ["up", "-c", str(config)])

    # Assert
    assert result.exit_code == 0, result.output

    load_mock.assert_called_once()
    # Default replicas should be None when not passed
    assert load_mock.call_args.kwargs.get("replicas") is None

    swarm_cls.assert_called_once()
    # Either check type:
    assert isinstance(swarm_cls.call_args.kwargs["cfg"], DummyCfg)
    # Or check identity with the exact object returned by the loader:
    assert swarm_cls.call_args.kwargs["cfg"] is cfg_instance

    # Context manager was used
    assert fake_swarm.__enter__.called
    assert fake_swarm.__exit__.called


@pytest.mark.parametrize("replicas", [1, 3, 8])
def test_cli_up_passes_replicas_override(tmp_path, mocker, replicas):
    config = tmp_path / "config.yaml"
    config.write_text("fake_yaml: true")

    class DummyCfg:
        pass

    from domyn_swarm.cli import main

    load_mock = mocker.patch.object(main, "_load_swarm_config", return_value=DummyCfg())
    fake_swarm = mocker.MagicMock()
    fake_swarm.__enter__.return_value = fake_swarm
    fake_swarm.__exit__.return_value = None
    mocker.patch.object(main, "DomynLLMSwarm", return_value=fake_swarm)

    runner = CliRunner()
    result = runner.invoke(main.app, ["up", "-c", str(config), "-r", str(replicas)])

    assert result.exit_code == 0, result.output
    assert load_mock.call_args.kwargs.get("replicas") == replicas


def test_up_prints_only_name(monkeypatch):
    class DummySwarm:
        name = "swarm-abc123"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    runner = CliRunner()
    monkeypatch.setattr(main, "_load_swarm_config", lambda *a, **k: object())
    monkeypatch.setattr(main, "DomynLLMSwarm", lambda cfg: DummySwarm())

    result = runner.invoke(main.app, ["up", "-c", "-"], input="{}")
    assert result.exit_code == 0
    assert result.output.strip() == "swarm-abc123"


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
    name = "my-swarm"
    _FakeStateManager._swarm_instance = _DummySwarm(name=name)

    result = runner.invoke(app, ["down", name, "--yes"])

    assert result.exit_code == 0, result.output
    # Success message printed
    assert "✅ Swarm my-swarm shutdown request sent." in result.output

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

    _FakeStateManager._swarm_instance = OrderedSwarm()

    result = runner.invoke(app, ["down", "swarm-x", "--yes"])
    assert result.exit_code == 0, result.output

    calls = _FakeStateManager._swarm_instance.calls
    # First a load tuple, then 'down', then 'delete_record'
    # We only assert relative order of down vs delete_record
    assert "down" in calls
    assert "delete_record" in calls
    assert calls.index("down") < calls.index("delete_record")


def test_down_by_name_running_user_declines_aborts(mocker):
    """When RUNNING and user declines confirm, we abort and do NOT call down()."""
    # Patch state manager load to return a running stub
    stub = _DummySwarm(status_phase="RUNNING", name="s1")
    mocker.patch.object(main, "SwarmStateManager", autospec=True)
    main.SwarmStateManager.load.return_value = stub  # type: ignore[attr-defined]

    runner = CliRunner()
    # Invoke via CLI to go through prompt inside down_by_name
    result = runner.invoke(main.app, ["down", "s1"], input="n\n")

    # Typer.Exit() without code -> exit_code == 0
    assert result.exit_code == 0
    assert "Aborting shutdown." in result.output
    assert not stub.down_called


def test_down_by_name_running_user_confirms_executes(mocker):
    """When RUNNING and user confirms, down() is called and success message printed."""
    stub = _DummySwarm(status_phase="RUNNING", name="s1")
    mocker.patch.object(main, "SwarmStateManager", autospec=True)
    main.SwarmStateManager.load.return_value = stub  # type: ignore[attr-defined]

    runner = CliRunner()
    result = runner.invoke(main.app, ["down", "s1"], input="y\n")

    assert result.exit_code == 0, result.output
    assert "✅ Swarm s1 shutdown request sent." in result.output
    assert stub.down_called


def test_down_with_config_no_matches_prints_message_and_exits0(tmp_path, mocker):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("name: base")

    mocker.patch.object(main, "_load_swarm_config", return_value=_DummyCfg("base"))
    mocker.patch.object(main, "SwarmStateManager", autospec=True)
    main.SwarmStateManager.list_by_base_name.return_value = []  # type: ignore[attr-defined]

    runner = CliRunner()
    result = runner.invoke(main.app, ["down", "-c", str(cfg_path)])

    assert result.exit_code == 0
    assert "No swarms found for base name 'base'." in result.output


def test_down_with_config_single_match_yes_skips_prompt_and_calls_down(
    tmp_path, mocker
):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("name: base")

    mocker.patch.object(main, "_load_swarm_config", return_value=_DummyCfg("base"))
    mocker.patch.object(main, "SwarmStateManager", autospec=True)
    main.SwarmStateManager.list_by_base_name.return_value = ["base-abc"]  # type: ignore[attr-defined]

    # Spy on down_by_name to ensure call (and that --yes is honored)
    called = {}

    def _spy_down_by_name(name: str, yes: bool):
        called["name"] = name
        called["yes"] = yes

    mocker.patch.object(main, "down_by_name", side_effect=_spy_down_by_name)

    runner = CliRunner()
    result = runner.invoke(main.app, ["down", "-c", str(cfg_path), "--yes"])

    assert result.exit_code == 0, result.output
    assert "Found 1 match: base-abc" in result.output
    assert called == {"name": "base-abc", "yes": True}


def test_down_with_config_single_match_user_declines_aborts(tmp_path, mocker):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("name: base")

    mocker.patch.object(main, "_load_swarm_config", return_value=_DummyCfg("base"))
    mocker.patch.object(main, "SwarmStateManager", autospec=True)
    main.SwarmStateManager.list_by_base_name.return_value = ["base-abc"]  # type: ignore[attr-defined]

    # Make sure we do not call down_by_name on decline
    mocker.patch.object(main, "down_by_name")

    runner = CliRunner()
    # decline confirmation
    result = runner.invoke(main.app, ["down", "-c", str(cfg_path)], input="n\n")

    # Typer.Abort() -> non-zero
    assert result.exit_code != 0
    main.down_by_name.assert_not_called()


def test_down_with_config_all_prompts_then_calls_all(tmp_path, mocker):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("name: base")

    mocker.patch.object(main, "_load_swarm_config", return_value=_DummyCfg("base"))
    mocker.patch.object(main, "SwarmStateManager", autospec=True)
    main.SwarmStateManager.list_by_base_name.return_value = ["a1", "a2"]  # type: ignore[attr-defined]

    mocker.patch.object(main, "down_by_name")

    runner = CliRunner()
    result = runner.invoke(
        main.app, ["down", "-c", str(cfg_path), "--all"], input="y\n"
    )

    assert result.exit_code == 0, result.output
    # Called for both with yes=True
    main.down_by_name.assert_any_call(name="a1", yes=True)
    main.down_by_name.assert_any_call(name="a2", yes=True)
    assert main.down_by_name.call_count == 2


def test_down_with_config_all_decline_aborts(tmp_path, mocker):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("name: base")

    mocker.patch.object(main, "_load_swarm_config", return_value=_DummyCfg("base"))
    mocker.patch.object(main, "SwarmStateManager", autospec=True)
    main.SwarmStateManager.list_by_base_name.return_value = ["a1", "a2"]  # type: ignore[attr-defined]

    mocker.patch.object(main, "down_by_name")

    runner = CliRunner()
    result = runner.invoke(
        main.app, ["down", "-c", str(cfg_path), "--all"], input="n\n"
    )

    assert result.exit_code != 0
    main.down_by_name.assert_not_called()


def test_down_with_config_select_picks_one_and_confirms(tmp_path, mocker):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("name: base")

    mocker.patch.object(main, "_load_swarm_config", return_value=_DummyCfg("base"))
    mocker.patch.object(main, "SwarmStateManager", autospec=True)
    main.SwarmStateManager.list_by_base_name.return_value = ["a1", "a2", "a3"]  # type: ignore[attr-defined]
    mocker.patch.object(main, "_pick_one", return_value="a2")
    mocker.patch.object(main, "down_by_name")

    runner = CliRunner()
    result = runner.invoke(
        main.app, ["down", "-c", str(cfg_path), "--select"], input="y\n"
    )

    assert result.exit_code == 0, result.output
    main.down_by_name.assert_called_once_with(name="a2", yes=True)


# ---------- Fallback (no name, no config) ----------


def test_down_fallback_to_last_swarm_success(mocker):
    mocker.patch.object(main, "SwarmStateManager", autospec=True)
    main.SwarmStateManager.get_last_swarm_name.return_value = "last-one"  # type: ignore[attr-defined]

    mocker.patch.object(main, "down_by_name")

    runner = CliRunner()
    result = runner.invoke(main.app, ["down"])

    assert result.exit_code == 0, result.output
    main.down_by_name.assert_called_once_with(name="last-one", yes=False)


def test_down_fallback_to_last_swarm_missing_exits_1(mocker):
    mocker.patch.object(main, "SwarmStateManager", autospec=True)
    main.SwarmStateManager.get_last_swarm_name.return_value = None  # type: ignore[attr-defined]

    runner = CliRunner()
    result = runner.invoke(main.app, ["down"])

    assert result.exit_code == 1
    assert "No swarms found to shut down." in result.output
