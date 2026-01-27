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

from __future__ import annotations

from rich.console import Console
from typer.testing import CliRunner

import domyn_swarm.cli.swarm as SW
from domyn_swarm.platform.protocols import ServingPhase, ServingStatus

COMMAND_NAME = "domyn-swarm swarm"


def _rec(name="alpha", platform="slurm", endpoint="http://alpha:9000"):
    return {"name": name, "platform": platform, "endpoint": endpoint}


def _fake_swarm_with_status(status: ServingStatus):
    class _Serving:
        def status(self, handle):
            return status

    class _Dep:
        serving = _Serving()

    class _Swarm:
        _deployment = _Dep()
        serving_handle = object()

    return _Swarm()


def test_iter_summaries_no_probe_uses_state_only(mocker):
    mocker.patch(
        "domyn_swarm.core.state.state_manager.SwarmStateManager.list_all",
        return_value=[_rec("a", "slurm", "http://a:9000"), _rec("b", "lepton", "")],
    )
    spy_from_state = mocker.patch("domyn_swarm.DomynLLMSwarm.from_state")

    out = list(SW._iter_summaries(probe=False))
    assert [s.name for s in out] == ["a", "b"]
    assert [s.backend for s in out] == ["slurm", "lepton"]
    assert [s.phase for s in out] == [
        "UNPROBED",
        "UNPROBED",
    ]
    assert [s.url for s in out] == ["http://a:9000", ""]
    spy_from_state.assert_not_called()


def test_iter_summaries_probe_success_populates_fields(mocker):
    mocker.patch(
        "domyn_swarm.core.state.state_manager.SwarmStateManager.list_all",
        return_value=[_rec("alpha", "slurm", "")],
    )
    st = ServingStatus(
        phase=ServingPhase.RUNNING,
        url="http://live:9000",
        detail={
            "http": 200,
            "rep": "RUNNING",
            "lb": "RUNNING",
            "raw_state": "Ready",
            "other": "x",
        },
    )
    mocker.patch("domyn_swarm.DomynLLMSwarm.from_state", return_value=_fake_swarm_with_status(st))

    out = list(SW._iter_summaries(probe=True))
    assert len(out) == 1
    s = out[0]
    assert s.name == "alpha"
    assert s.backend == "slurm"
    assert s.phase == "RUNNING"
    assert s.url == "http://live:9000"
    assert s.http == 200
    assert s.extra == {"rep": "RUNNING", "lb": "RUNNING", "raw_state": "Ready"}


def test_iter_summaries_probe_skips_unknown(mocker):
    mocker.patch(
        "domyn_swarm.core.state.state_manager.SwarmStateManager.list_all",
        return_value=[_rec("dead", "slurm", "")],
    )
    st = ServingStatus(phase=ServingPhase.UNKNOWN, url=None, detail=None)
    mocker.patch("domyn_swarm.DomynLLMSwarm.from_state", return_value=_fake_swarm_with_status(st))

    out = list(SW._iter_summaries(probe=True))
    assert out == []


def test_iter_summaries_probe_errors_skip_record(mocker):
    mocker.patch(
        "domyn_swarm.core.state.state_manager.SwarmStateManager.list_all",
        return_value=[_rec("oops", "lepton", "http://rec:9000")],
    )
    mocker.patch("domyn_swarm.DomynLLMSwarm.from_state", side_effect=RuntimeError("boom"))

    out = list(SW._iter_summaries(probe=True))
    assert out == []


def test_iter_summaries_probe_skips_failed_or_stopped(mocker):
    mocker.patch(
        "domyn_swarm.core.state.state_manager.SwarmStateManager.list_all",
        return_value=[
            _rec("failed", "slurm", ""),
            _rec("stopped", "lepton", ""),
            _rec("ok", "slurm", ""),
        ],
    )
    statuses = {
        "failed": ServingStatus(phase=ServingPhase.FAILED, url=None, detail=None),
        "stopped": ServingStatus(phase=ServingPhase.STOPPED, url=None, detail=None),
        "ok": ServingStatus(phase=ServingPhase.RUNNING, url="http://ok", detail=None),
    }

    def _from_state(*, deployment_name: str):
        return _fake_swarm_with_status(statuses[deployment_name])

    mocker.patch("domyn_swarm.DomynLLMSwarm.from_state", side_effect=_from_state)

    out = list(SW._iter_summaries(probe=True))
    assert [s.name for s in out] == ["ok"]


def test_cli_list_prints_no_swarms_message(mocker):
    mocker.patch.object(SW, "_iter_summaries", return_value=[])
    runner = CliRunner()
    # Provide a prog_name so Click doesn't try to read cli.name from Typer
    result = runner.invoke(SW.swarm_app, ["list", "--no-probe"])
    print(result.output)
    assert result.exit_code == 0
    assert "No swarms found." in result.output


def test_cli_list_calls_renderer_with_rows_and_default_probe(mocker):
    seen = {}
    fake_rows = [
        SW.SwarmSummary(
            name="a",
            backend="slurm",
            phase="RUNNING",
            url="http://a",
            http=200,
            extra={"rep": "RUNNING"},
        )
    ]

    def fake_iter_summaries(*, probe):
        seen["probe"] = probe
        return fake_rows

    mocker.patch.object(SW, "_iter_summaries", side_effect=fake_iter_summaries)
    render_mock = mocker.patch("domyn_swarm.cli.tui.list_view.render_swarm_list")

    runner = CliRunner()
    result = runner.invoke(SW.swarm_app, ["list"])
    assert result.exit_code == 0
    assert seen.get("probe") is True

    render_mock.assert_called_once()
    args, kwargs = render_mock.call_args
    assert args[0] == fake_rows
    assert isinstance(kwargs.get("console"), Console)


def test_cli_list_no_probe_flag_disables_probe(mocker):
    seen = {}
    mocker.patch.object(
        SW,
        "_iter_summaries",
        side_effect=lambda *, probe: seen.setdefault("probe", probe)
        or [SW.SwarmSummary("n", "slurm", "UNKNOWN")],
    )
    mocker.patch("domyn_swarm.cli.tui.list_view.render_swarm_list")

    runner = CliRunner()
    result = runner.invoke(SW.swarm_app, ["list", "--no-probe"])
    assert result.exit_code == 0
    assert seen.get("probe") is False
