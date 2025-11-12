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

import pathlib

from typer.testing import CliRunner
import yaml

from domyn_swarm.cli.pool import pool_app
from domyn_swarm.config.pool import SwarmPoolElement


def test_deploy_pool_parses_yaml(monkeypatch, tmp_path):
    runner = CliRunner()

    with runner.isolated_filesystem(tmp_path):
        # Prepare files in current working dir
        pool_entries = [
            SwarmPoolElement(name="swarm-a", config_path="a.yaml"),
            SwarmPoolElement(name="swarm-b", config_path="b.yaml"),
        ]
        pool_path = tmp_path / pathlib.Path("pool.yaml")

        for element in pool_entries:
            (tmp_path / element.config_path).write_text("dummy_config: true\n")

        dumps = [element.model_dump() for element in pool_entries]
        pool_path.write_text(yaml.safe_dump({"pool": dumps}))

        # Capture what gets passed to the context manager
        captured = {}

        class DummySwarmPoolConfig: ...

        class DummySwarmConfig:
            @staticmethod
            def read(path):
                return f"cfg@{path}"

        def dummy_swarm(cfg):
            print(f"🧠 Constructing DomynLLMSwarm({cfg=})")
            return cfg

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
                {
                    "pool": [
                        SwarmPoolElement(name="swarm-a", config_path="a.yaml"),
                        SwarmPoolElement(name="swarm-b", config_path="b.yaml"),
                    ]
                },
            )(),
        )
        monkeypatch.setattr("domyn_swarm.cli.pool.DomynLLMSwarmConfig", DummySwarmConfig)
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

        expected = (("cfg@a.yaml"), ("cfg@b.yaml"))
        assert captured["args"] == expected
