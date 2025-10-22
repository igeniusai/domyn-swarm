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
import json

from pydantic import BaseModel

import domyn_swarm.cli.swarm as SW


class DummyCfg(BaseModel):
    foo: int = 42
    bar: str = "baz"


class FakeSwarm:
    def __init__(self, platform="slurm", endpoint="http://host:9000", cfg=None):
        self._platform = platform
        self.endpoint = endpoint
        self.cfg = cfg or DummyCfg()


def _patch_state_load(mocker, swarm: FakeSwarm):
    return mocker.patch.object(
        SW, "SwarmStateManager", autospec=True
    ).load.return_value.__setattr__(
        "__wrapped__", swarm
    )  # just to keep type linters calm


def test_describe_table_calls_renderer_with_expected_payload(mocker):
    # Arrange: fake state + renderer spy
    swarm = FakeSwarm(
        platform="slurm", endpoint="http://ep", cfg=DummyCfg(foo=7, bar="x")
    )
    # Patch the state loader to return our fake swarm
    mock_sm = mocker.patch.object(SW, "SwarmStateManager", autospec=True)
    mock_sm.load.return_value = swarm
    # Patch the renderer
    render = mocker.patch("domyn_swarm.cli.tui.describe_view.render_swarm_description")

    # Act
    SW.describe_swarm(name="my-swarm", output="table")

    # Assert
    render.assert_called_once()
    call = render.call_args
    assert call.kwargs["name"] == "my-swarm"
    assert call.kwargs["backend"] == "slurm"
    assert call.kwargs["endpoint"] == "http://ep"
    # cfg passed to renderer is already a dict from model_dump
    assert call.kwargs["cfg"] == {"foo": 7, "bar": "x"}
    # console instance is provided
    assert call.kwargs["console"] is not None


def test_describe_yaml_outputs_valid_yaml(mocker):
    import yaml

    swarm = FakeSwarm(platform="lepton", endpoint="", cfg=DummyCfg(foo=1, bar="b"))
    mock_sm = mocker.patch.object(SW, "SwarmStateManager", autospec=True)
    mock_sm.load.return_value = swarm

    echo = mocker.patch.object(SW.typer, "echo")

    SW.describe_swarm(name="lepton-01", output="yaml")

    echo.assert_called_once()
    yaml_text = echo.call_args.args[0]
    data = yaml.safe_load(yaml_text)

    assert data["name"] == "lepton-01"
    assert data["backend"] == "lepton"
    assert data["endpoint"] == ""
    assert data["config"] == {"foo": 1, "bar": "b"}


def test_describe_json_outputs_valid_json(mocker):
    swarm = FakeSwarm(platform="slurm", endpoint="http://ep", cfg=DummyCfg(foo=99))
    mock_sm = mocker.patch.object(SW, "SwarmStateManager", autospec=True)
    mock_sm.load.return_value = swarm

    echo = mocker.patch.object(SW.typer, "echo")

    SW.describe_swarm(name="s1", output="json")

    echo.assert_called_once()
    js = echo.call_args.args[0]
    data = json.loads(js)

    assert data["name"] == "s1"
    assert data["backend"] == "slurm"
    assert data["endpoint"] == "http://ep"
    assert data["config"] == {"foo": 99, "bar": "baz"}  # default "bar" applies


def test_describe_output_is_case_insensitive(mocker):
    swarm = FakeSwarm(platform="slurm", endpoint=None, cfg=DummyCfg())
    mock_sm = mocker.patch.object(SW, "SwarmStateManager", autospec=True)
    mock_sm.load.return_value = swarm

    echo = mocker.patch.object(SW.typer, "echo")

    SW.describe_swarm(name="s2", output="YAML")  # mixed-case
    echo.assert_called_once()


def test_describe_invalid_output_falls_back_to_table(mocker):
    swarm = FakeSwarm(platform="slurm", endpoint="http://ep", cfg=DummyCfg())
    mock_sm = mocker.patch.object(SW, "SwarmStateManager", autospec=True)
    mock_sm.load.return_value = swarm

    render = mocker.patch("domyn_swarm.cli.tui.describe_view.render_swarm_description")

    SW.describe_swarm(name="s3", output="not-a-mode")

    render.assert_called_once()
