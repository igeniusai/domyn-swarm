from typer.testing import CliRunner

from domyn_swarm.cli.main import app

runner = CliRunner()


def test_cli_version(monkeypatch):
    monkeypatch.setattr("domyn_swarm.cli.main.metadata.version", lambda _: "1.2.3")

    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "domyn-swarm CLI Version: " in result.stdout
    assert "1.2.3" in result.stdout


def test_cli_up_requires_config(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("fake_yaml: true")

    def dummy_load_config(*args, **kwargs):
        class DummyCfg:
            replicas = 1

        return DummyCfg()

    def dummy_start_swarm(name, cfg, reverse_proxy):
        assert name is None
        assert cfg.replicas == 1
        assert reverse_proxy is False

    from domyn_swarm.cli import main

    monkeypatch.setattr(main, "_load_swarm_config", dummy_load_config)
    monkeypatch.setattr(main, "_start_swarm", dummy_start_swarm)

    result = runner.invoke(app, ["up", "-c", str(config)])
    assert result.exit_code == 0
