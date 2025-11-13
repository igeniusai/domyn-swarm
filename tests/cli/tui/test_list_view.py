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

import types

from rich.console import Console
from rich.padding import Padding
from rich.style import Style
from rich.text import Text

import domyn_swarm.cli.tui.list_view as list_view


class FakeTable:
    """Minimal table stub that collects rows so we can assert on them."""

    def __init__(self):
        self.columns = None
        self.rows = []

    def add_row(self, *cells):
        self.rows.append(cells)


def _mk_row(**kwargs):
    """Simple row object with attribute access."""
    obj = types.SimpleNamespace()
    for k, v in kwargs.items():
        setattr(obj, k, v)
    return obj


def _span_has_link_style(t: Text, url: str) -> bool:
    """Check if any span in Text applies a link style for the given URL."""
    for span in t.spans:
        st = span.style
        # Support both Style and str variants across Rich versions
        if isinstance(st, str):
            if f"link {url}" in st:
                return True
        else:
            # Style object
            if isinstance(st, Style) and getattr(st, "link", None) == url:
                return True
    return False


def test_render_swarm_list_builds_rows_and_prints(mocker):
    # Arrange
    fake_table = FakeTable()
    # Patch imported helpers inside the module under test
    mock_list_table = mocker.patch.object(list_view, "list_table", return_value=fake_table)
    mock_phase_badge = mocker.patch.object(
        list_view, "phase_badge", side_effect=lambda p: Text(f"PH-{p}")
    )
    mock_http_badge = mocker.patch.object(
        list_view,
        "http_badge",
        side_effect=lambda h: Text("200 OK") if h == 200 else None,
    )

    console = mocker.Mock(spec=Console)

    rows = [
        _mk_row(
            name="alpha",
            backend="slurm",
            phase="RUNNING",
            url="http://host:9000",
            http=200,
            extra={"rep": "RUNNING", "raw_state": "Ready", "ignored": "x"},
        ),
        _mk_row(
            name="beta",
            backend="lepton",
            phase="PENDING",
            url=None,
            http=None,
            extra=None,
        ),
    ]

    # Act
    list_view.render_swarm_list(rows, console=console)

    # Assert list_table was called with expected columns
    mock_list_table.assert_called_once()
    args, kwargs = mock_list_table.call_args
    expected_cols = [" Name", "Backend", "Phase", "Endpoint", "Notes"]
    if "columns" in kwargs:
        assert kwargs["columns"] == expected_cols
    else:
        # fallback in case implementation switches to positional later
        assert args == (expected_cols,)

    # Assert console.print was called with our fake table
    console.print.assert_called_once_with(fake_table)

    # We expect two rows added
    assert len(fake_table.rows) == 2

    # ---- Row 0 checks ----
    r0 = fake_table.rows[0]
    # name is left-padded via Padding
    assert isinstance(r0[0], Padding)
    assert r0[0].renderable == "alpha"
    assert (r0[0].top, r0[0].right, r0[0].bottom, r0[0].left) == (0, 0, 0, 1)

    # backend uppercased
    assert r0[1] == "SLURM"

    # phase cell is what phase_badge returned
    assert isinstance(r0[2], Text)
    assert r0[2].plain == "PH-RUNNING"
    mock_phase_badge.assert_any_call("RUNNING")

    # url cell is Text with link style applied
    assert isinstance(r0[3], Text)
    assert r0[3].plain == "http://host:9000"
    assert _span_has_link_style(r0[3], "http://host:9000")

    # notes: "200 OK, rep=RUNNING, raw_state=Ready"
    assert isinstance(r0[4], str)
    assert r0[4] == "200 OK, rep=RUNNING, raw_state=Ready"
    mock_http_badge.assert_any_call(200)

    # ---- Row 1 checks ----
    r1 = fake_table.rows[1]
    assert isinstance(r1[0], Padding)
    assert r1[0].renderable == "beta"
    assert r1[1] == "LEPTON"
    assert isinstance(r1[2], Text)
    assert r1[2].plain == "PH-PENDING"
    mock_phase_badge.assert_any_call("PENDING")

    # url absent -> "—" and no link style
    assert isinstance(r1[3], Text)
    assert r1[3].plain == "—"
    assert r1[3].spans == []  # no link styling

    # no http badge and no extras -> "—"
    assert r1[4] == "—"


def test_render_swarm_list_handles_missing_fields(mocker):
    fake_table = FakeTable()
    mocker.patch.object(list_view, "list_table", return_value=fake_table)
    mocker.patch.object(list_view, "phase_badge", side_effect=lambda p: Text(f"PH-{p}"))
    mocker.patch.object(list_view, "http_badge", return_value=None)

    console = mocker.Mock(spec=Console)

    # Row missing many fields → defaults kick in
    rows = [
        _mk_row(name="gamma")  # backend/phase/url/http/extra absent
    ]

    list_view.render_swarm_list(rows, console=console)

    assert len(fake_table.rows) == 1
    r = fake_table.rows[0]

    # name present, padded
    assert isinstance(r[0], Padding)
    assert r[0].renderable == "gamma"

    # backend defaults to "—" then uppercased (still "—")
    assert r[1] == "—"

    # phase defaults to "UNKNOWN"
    assert isinstance(r[2], Text)
    assert r[2].plain == "PH-UNKNOWN"

    # url defaults to "—"
    assert isinstance(r[3], Text)
    assert r[3].plain == "—"

    # notes defaults to "—"
    assert r[4] == "—"

    console.print.assert_called_once_with(fake_table)
