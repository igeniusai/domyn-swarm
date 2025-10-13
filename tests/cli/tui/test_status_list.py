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

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel

from domyn_swarm.cli.tui import status_list as SL
from domyn_swarm.platform.protocols import ServingPhase, ServingStatus


def _mk_status(phase="RUNNING", url="http://host:9000", detail=None):
    return ServingStatus(phase=ServingPhase(phase), url=url, detail=detail or {})


def test_render_multi_status_builds_panels_and_prints(mocker):
    # Patch helper renderers to be deterministic
    mocker.patch.object(SL, "_phase_badge", side_effect=lambda p: f"PH({p})")
    mocker.patch.object(SL, "_fmt_http", side_effect=lambda d: "HTTP-OK" if d else None)
    mocker.patch.object(SL, "_color_state", side_effect=lambda x: f"C({x})")

    console = mocker.Mock(spec=Console)

    items = [
        (
            "alpha",
            "slurm",
            _mk_status(
                phase="RUNNING",
                url="http://alpha:9000",
                detail={
                    "http": 200,
                    "rep": "RUNNING",
                    "lb": "RUNNING",
                    "raw_state": "Ready",
                },
            ),
        ),
        (
            "beta",
            "lepton",
            _mk_status(
                phase="PENDING",
                url=None,
                detail={"raw_state": "Stopped"},
            ),
        ),
    ]

    SL.render_multi_status(items, console=console)

    # printed once with a Columns object
    console.print.assert_called_once()
    (arg,), _ = console.print.call_args
    assert isinstance(arg, Columns)

    # It should render one panel per item
    renderables = list(arg.renderables)  # public attr on Columns
    assert len(renderables) == len(items)
    assert all(isinstance(p, Panel) for p in renderables)

    # Panel titles should include name and UPPER backend
    titles = [str(p.title) for p in renderables]
    assert any("alpha" in t and "SLURM" in t for t in titles)
    assert any("beta" in t and "LEPTON" in t for t in titles)


def test_render_multi_status_text_content_smoke(mocker):
    # Make helpers return simple strings Rich can render
    mocker.patch.object(SL, "_phase_badge", side_effect=lambda p: f"PH({p})")
    mocker.patch.object(SL, "_fmt_http", side_effect=lambda d: "HTTP-OK" if d else None)
    mocker.patch.object(SL, "_color_state", side_effect=lambda x: f"C({x})")

    console = Console(record=True, width=120)

    items = [
        (
            "alpha",
            "slurm",
            _mk_status(
                phase="RUNNING",
                url="http://alpha:9000",
                detail={
                    "http": 200,
                    "rep": "RUNNING",
                    "lb": "RUNNING",
                    "raw_state": "Ready",
                },
            ),
        ),
        (
            "beta",
            "lepton",
            _mk_status(phase="PENDING", url=None, detail={"raw_state": "Stopped"}),
        ),
    ]

    SL.render_multi_status(items, console=console)
    out = console.export_text()

    # Names / backends
    assert "alpha" in out and "SLURM" in out
    assert "beta" in out and "LEPTON" in out

    # Phase badge content
    assert "PH(ServingPhase.RUNNING)" in out
    assert "PH(ServingPhase.PENDING)" in out

    # Endpoint URLs (clickable links become plain in text export; at least ensure text shows)
    assert "http://alpha:9000" in out
    # beta has no URL, table should show a placeholder dash
    assert "Endpoint" in out

    # HTTP row
    assert "HTTP" in out and "HTTP-OK" in out

    # Replica/LB & Platform rows via _color_state stubs
    assert "Replica" in out and "C(RUNNING)" in out
    # In this compact view, LB is labeled as "Endpoint" per implementation
    assert "Endpoint" in out
    assert "Platform" in out and ("C(Ready)" in out or "C(Stopped)" in out)


def test_render_multi_status_no_http_or_detail(mocker):
    mocker.patch.object(SL, "_phase_badge", side_effect=lambda p: f"PH({p})")
    mocker.patch.object(SL, "_fmt_http", side_effect=lambda d: None)
    mocker.patch.object(SL, "_color_state", side_effect=lambda x: f"C({x})")

    console = Console(record=True, width=100)

    items = [
        ("solo", "slurm", _mk_status(phase="RUNNING", url=None, detail=None)),
    ]

    SL.render_multi_status(items, console=console)
    out = console.export_text()

    # No HTTP row text when no detail/http
    assert "HTTP" not in out
    # Still shows Phase/Endpoint labels
    assert "Phase" in out and "Endpoint" in out
