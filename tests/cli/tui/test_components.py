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

from types import SimpleNamespace

import pytest
from rich.style import Style
from rich.table import Table
from rich.text import Text

from domyn_swarm.cli.tui import components as C


def _effective_style(txt: Text) -> Style | None:
    """Return the effective style of a Text: span style if present, else default style."""
    if txt.spans:
        # Take the first span style (what .stylize produces)
        return txt.spans[0].style
    return txt.style


def _style_equals(txt: Text, expected: str) -> bool:
    """Compare a Text's effective style to an expected style string."""
    sty = _effective_style(txt)
    if sty is None:
        return False
    # Normalize both sides via Style.parse; some Rich versions store style as str
    return Style.parse(str(sty)) == Style.parse(expected)


# ---------- _phase_badge -----------------------------------------------------


def test_phase_badge_uses_glyph_and_style(mocker):
    # Patch the constants inside the module under test
    mocker.patch.object(C, "_PHASE_GLYPH", {"RUNNING": "*"})
    mocker.patch.object(C, "PHASE_STYLE", {"RUNNING": "bold blue"})
    phase_like = SimpleNamespace(value="RUNNING")

    t: Text = C._phase_badge(phase_like)

    assert isinstance(t, Text)
    assert t.plain == "* RUNNING"
    # style applied via .stylize() → span-based
    assert _style_equals(t, "bold blue")


def test_phase_badge_unknown_uses_fallbacks(mocker):
    mocker.patch.object(C, "_PHASE_GLYPH", {})  # force fallback glyph
    mocker.patch.object(C, "PHASE_STYLE", {})  # force fallback style
    phase_like = SimpleNamespace(value="WHATEVER")

    t: Text = C._phase_badge(phase_like)

    assert t.plain == "• WHATEVER"
    assert _style_equals(t, "bold white on grey23")


# ---------- _color_state ------------------------------------------------------


@pytest.mark.parametrize(
    "state, bad_set, wait_set, expected_style",
    [
        ("FAILED", {"FAILED"}, set(), "bold red"),
        ("PENDING", set(), {"PENDING"}, "bold yellow"),
        ("running", set(), set(), "bold green"),  # case-insensitive
        ("READY", set(), set(), "bold green"),
        ("mystery", set(), set(), "bold cyan"),
        (None, set(), set(), "bold cyan"),  # None -> "UNKNOWN"
    ],
)
def test_color_state_categories(mocker, state, bad_set, wait_set, expected_style):
    mocker.patch.object(C, "_BAD_STATES", set(bad_set))
    mocker.patch.object(C, "_WAIT_STATES", set(wait_set))

    txt = C._color_state(state)

    assert isinstance(txt, Text)
    # _color_state uses Text(style="…"), so default style, no spans
    assert _style_equals(txt, expected_style)


# ---------- _fmt_http ---------------------------------------------------------


def test_fmt_http_none_detail_returns_none():
    assert C._fmt_http(None) is None


def test_fmt_http_missing_http_key_returns_none():
    assert C._fmt_http({"something": 1}) is None


def test_fmt_http_200_ok():
    out = C._fmt_http({"http": 200})
    assert isinstance(out, Text)
    assert out.plain == "200 OK"
    # constructed with style="bold green"
    assert _style_equals(out, "bold green")


@pytest.mark.parametrize("val", ["unready", "timeout", "UnReAdY", "TIMEOUT"])
def test_fmt_http_transient_yellow(val):
    out = C._fmt_http({"http": val})
    assert isinstance(out, Text)
    assert out.plain.lower() == str(val).lower()
    assert _style_equals(out, "bold yellow")


def test_fmt_http_other_red():
    out = C._fmt_http({"http": 502})
    assert isinstance(out, Text)
    assert out.plain == "502"
    assert _style_equals(out, "bold red")


# ---------- _add_if -----------------------------------------------------------


def _mk_table() -> Table:
    t = Table.grid(padding=(0, 1))
    t.add_column("Field", style="bold dim", no_wrap=True, justify="right")
    t.add_column("Value", overflow="fold")
    return t


def test_add_if_does_nothing_on_none():
    tbl = _mk_table()
    C._add_if(tbl, "Label", None)
    assert len(tbl.rows) == 0


def test_add_if_adds_text_directly():
    tbl = _mk_table()
    C._add_if(tbl, "Phase", Text("RUNNING", style="green"))
    assert len(tbl.rows) == 1  # one row added


def test_add_if_coerces_str_to_text():
    tbl = _mk_table()
    C._add_if(tbl, "HTTP", "200 OK")
    assert len(tbl.rows) == 1
