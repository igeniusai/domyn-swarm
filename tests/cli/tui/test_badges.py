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

import pytest
from rich.style import Style
from rich.text import Text

import domyn_swarm.cli.tui.badges as badges


def _norm_style(s):
    # Rich may return a Style or a str; normalize to Style for comparisons.
    return Style.parse(s) if isinstance(s, str) else s


def test_phase_badge_uses_theme_helpers(mocker):
    # Patch the imported helpers inside the badges module
    mock_glyph = mocker.patch.object(badges, "phase_glyph", return_value="~")
    mock_style = mocker.patch.object(badges, "phase_style", return_value="bold magenta")

    t: Text = badges.phase_badge("PENDING")

    mock_glyph.assert_called_once_with("PENDING")
    mock_style.assert_called_once_with("PENDING")

    assert isinstance(t, Text)
    assert t.plain == "~ PENDING"  # glyph + space + phase
    # One style span covering the whole text
    assert len(t.spans) == 1
    span = t.spans[0]
    assert span.start == 0
    assert span.end == len(t.plain)
    assert _norm_style(span.style) == Style.parse("bold magenta")


def test_http_badge_none_returns_none():
    assert badges.http_badge(None) is None


def test_http_badge_200_ok_green():
    t = badges.http_badge(200)
    assert isinstance(t, Text)
    assert t.plain == "200 OK"
    assert len(t.spans) == 1
    assert _norm_style(t.spans[0].style) == Style.parse("bold green")


@pytest.mark.parametrize("val", ["unready", "TIMEOUT"])
def test_http_badge_unready_or_timeout_yellow(val):
    t = badges.http_badge(val)
    assert isinstance(t, Text)
    assert t.plain.lower() in {"unready", "timeout"}
    assert len(t.spans) == 1
    assert _norm_style(t.spans[0].style) == Style.parse("bold yellow")


def test_http_badge_other_codes_red():
    t = badges.http_badge(503)
    assert isinstance(t, Text)
    assert t.plain == "503"
    assert len(t.spans) == 1
    assert _norm_style(t.spans[0].style) == Style.parse("bold red")
