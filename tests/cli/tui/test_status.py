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

import re

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from domyn_swarm.cli.tui import status as STATUS
from domyn_swarm.platform.protocols import ServingPhase, ServingStatus


def _mk_status(phase="RUNNING", url="http://x:9", detail=None):
    return ServingStatus(phase=ServingPhase(phase), url=url, detail=detail or {})


def test_render_swarm_status_prints_panel(mocker):
    console = mocker.Mock(spec=Console)
    st = _mk_status(detail={"http": 200})

    # minimal patches to keep behavior simple
    mocker.patch.object(STATUS, "_phase_badge", return_value=Text("PHASE"))
    mocker.patch.object(STATUS, "_fmt_http", return_value=Text("HTTPOK"))

    STATUS.render_swarm_status("my-swarm", "slurm", st, console=console)

    console.print.assert_called_once()
    (arg,), _ = console.print.call_args
    assert isinstance(arg, Panel)

    # Panel title/subtitle sanity
    assert "my-swarm" in str(arg.title)
    assert "SLURM" in str(arg.title)  # backend uppercased
    # subtitle should be the URL
    assert arg.subtitle == st.url


def test_render_swarm_status_core_fields_in_text(mocker):
    # Make helper outputs deterministic to search for
    mocker.patch.object(STATUS, "_phase_badge", return_value=Text("PH-BADGE"))
    mocker.patch.object(STATUS, "_fmt_http", return_value=None)  # no HTTP row

    console = Console(record=True, width=100)
    st = _mk_status(phase="PENDING", url="http://host:9000", detail=None)

    STATUS.render_swarm_status("alpha", "lepton", st, console=console)
    out = console.export_text()

    # Banner contains name/backend
    assert "SWARM " in out
    assert "alpha" in out
    assert "LEPTON" in out

    # Phase badge text appears
    assert "PH-BADGE" in out

    # Endpoint link text present
    assert "http://host:9000" in out


def test_render_swarm_status_slurm_diag_rows_and_helper_calls(mocker):
    # Capture _color_state arguments, return predictable text
    calls = []

    def _cs(arg):
        calls.append(arg)
        return Text(f"C({arg})")

    mocker.patch.object(STATUS, "_phase_badge", return_value=Text("PH"))
    mocker.patch.object(STATUS, "_fmt_http", return_value=Text("HTTP-OK"))
    mocker.patch.object(STATUS, "_color_state", side_effect=_cs)

    console = Console(record=True, width=120)
    detail = {
        "rep": "RUNNING",
        "lb": "PENDING",
        "http": 200,
        "foo": "bar",  # should show up in Extras
    }
    st = _mk_status(phase="RUNNING", url="http://host:9", detail=detail)

    STATUS.render_swarm_status("slurm-a", "slurm", st, console=console)
    out = console.export_text()

    # HTTP row present via _fmt_http
    assert "HTTP" in out and "HTTP-OK" in out

    # Replica/LB rows colored via _color_state
    assert "Replica" in out and "C(RUNNING)" in out
    assert re.search(r"\bLB\b", out)
    assert "C(PENDING)" in out

    # Extras (foo=bar) printed, not including filtered keys
    assert "Extra" in out and "foo=bar" in out

    # Verify helper received expected args in order (rep, lb)
    assert calls[:2] == ["RUNNING", "PENDING"]


def test_render_swarm_status_lepton_platform_and_http_unready(mocker):
    mocker.patch.object(STATUS, "_phase_badge", return_value=Text("PHASE-X"))

    # Return Text for HTTP so it gets added via _add_if path
    mocker.patch.object(STATUS, "_fmt_http", return_value=Text("HTTP-UNREADY"))

    # Color state should be called with raw_state
    seen = {}

    def _cs(arg):
        seen["arg"] = arg
        return Text(f"C({arg})")

    mocker.patch.object(STATUS, "_color_state", side_effect=_cs)

    console = Console(record=True, width=100)
    detail = {"raw_state": "Ready", "http": "unready"}
    st = _mk_status(phase="INITIALIZING", url=None, detail=detail)

    STATUS.render_swarm_status("lepton-a", "lepton", st, console=console)
    out = console.export_text()

    # Platform row from raw_state
    assert "Platform" in out and "C(Ready)" in out
    # HTTP row
    assert "HTTP" in out and "HTTP-UNREADY" in out
    # No endpoint URL
    assert "Endpoint" in out  # label still present
    # Confirm helper saw the expected raw_state
    assert seen.get("arg") == "Ready"
