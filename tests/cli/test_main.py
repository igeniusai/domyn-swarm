# Copyright 2025 Domyn
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

from typer.testing import CliRunner

from domyn_swarm.cli.main import app

runner = CliRunner()


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
