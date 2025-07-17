import pathlib

import yaml
from typer.testing import CliRunner

from domyn_swarm.cli.pool import pool_app


def test_deploy_pool_parses_yaml(monkeypatch, tmp_path):
    runner = CliRunner()

    with runner.isolated_filesystem(tmp_path):
        # Prepare files in current working dir
        pool_entries = [
            ["swarm-a", "a.yaml"],
            ["swarm-b", "b.yaml"],
        ]
        pool_path = tmp_path / pathlib.Path("pool.yaml")

        for _, path in pool_entries:
            (tmp_path / path).write_text("dummy_config: true\n")

        pool_path.write_text(yaml.safe_dump({"pool": pool_entries}))

        # Capture what gets passed to the context manager
        captured = {}

        class DummySwarmPoolConfig:
            @staticmethod
            def model_validate(data):
                assert hasattr(data, "read"), "Expected a file-like object"
                loaded = yaml.safe_load(data.read())
                return type("Validated", (), {"pool": loaded["pool"]})()

        class DummySwarmConfig:
            @staticmethod
            def read(path):
                return f"cfg@{path}"

        def dummy_swarm(name, cfg):
            print(f"🧠 Constructing DomynLLMSwarm({name=}, {cfg=})")
            return (name, cfg)

        class DummyContext:
            def __init__(self, *args):
                print(f"✅ create_swarm_pool called with: {args}")
                captured["args"] = args

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

        # Patch dependencies
        monkeypatch.setattr(
            "domyn_swarm.cli.pool.SwarmPoolConfig.model_validate",
            lambda file_obj: type(
                "DummyPoolConfig",
                (),
                {"pool": [["swarm-a", "a.yaml"], ["swarm-b", "b.yaml"]]},
            )(),
        )
        monkeypatch.setattr(
            "domyn_swarm.cli.pool.DomynLLMSwarmConfig", DummySwarmConfig
        )
        monkeypatch.setattr("domyn_swarm.cli.pool.DomynLLMSwarm", dummy_swarm)
        monkeypatch.setattr(
            "domyn_swarm.cli.pool.create_swarm_pool", lambda *args: DummyContext(*args)
        )

        # Run CLI
        result = runner.invoke(pool_app, [str(pool_path)])
        if result.exit_code != 0:
            print("❌ Test failed unexpectedly")
            print("STDOUT:\n", result.stdout)
            print("STDERR:\n", result.stderr)
            print("Exception:\n", result.exception)

        assert result.exit_code == 0
        assert "args" in captured

        expected = (
            ("swarm-a", "cfg@a.yaml"),
            ("swarm-b", "cfg@b.yaml"),
        )
        assert captured["args"] == expected
