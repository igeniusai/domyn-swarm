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

from typing import Any, Optional

from rich.table import Table
from rich.text import Text

from domyn_swarm.platform.protocols import ServingPhase

from ..tui.theme import _BAD_STATES, _PHASE_GLYPH, _WAIT_STATES, PHASE_STYLE


def _phase_badge(phase: "ServingPhase") -> Text:
    """Create a styled text badge for a serving phase.

    Args:
        phase (ServingPhase): The serving phase to create a badge for.

    Returns:
        Text: A Rich Text object containing the phase name with emoji and styling.
            The emoji and style are determined by the phase value using internal
            mapping dictionaries.
    """
    s = str(phase.value)
    t = Text(f"{_PHASE_GLYPH.get(s, '•')} {s}")
    t.stylize(PHASE_STYLE.get(s, "bold white on grey23"))
    return t


def _color_state(state: Any) -> Text:
    """Colors a state string based on its value.

    This function takes a state value and returns a styled Text object with
    appropriate coloring based on the state type. Bad states are colored red,
    waiting states are yellow, running/ready states are green, and all other
    states are cyan.

    Args:
        state (Any): The state value to be colored. Can be any type that can be
            converted to string. If None, defaults to "UNKNOWN".

    Returns:
        Text: A Text object with the state string styled with appropriate colors:
            - Red (bold) for bad states (defined in _BAD_STATES)
            - Yellow (bold) for waiting states (defined in _WAIT_STATES)
            - Green (bold) for "RUNNING" and "READY" states
            - Cyan (bold) for all other states
    """
    s = str(state or "UNKNOWN").upper()
    if s in _BAD_STATES:
        return Text(s, style="bold red")
    if s in _WAIT_STATES:
        return Text(s, style="bold yellow")
    if s in {"RUNNING", "READY"}:
        return Text(s, style="bold green")
    return Text(s, style="bold cyan")


def _fmt_http(detail: dict[str, Any] | None) -> Optional[Text]:
    """Format HTTP status information for display in the TUI.

    Args:
        detail: Dictionary containing HTTP status information, may be None.

    Returns:
        A formatted Text object with appropriate styling based on the HTTP status,
        or None if no detail or HTTP information is provided.

        - HTTP 200: Green "200 OK" text
        - "unready" or "timeout": Yellow text
        - Other HTTP codes: Red text
    """
    if not detail:
        return None
    http = detail.get("http")
    if http is None:
        return None
    if http == 200:
        return Text("200 OK", style="bold green")
    if str(http).lower() in {"unready", "timeout"}:
        return Text(str(http), style="bold yellow")
    return Text(str(http), style="bold red")


def _add_if(details: Table, label: str, value: Optional[Text | str]):
    """Add a row to the table if the value is not None.

    Args:
        details (Table): The table to add the row to.
        label (str): The label for the row.
        value (Optional[Text | str]): The value to add. If None, no row is added.
            If a string, it will be converted to Text.
    """
    if value is None:
        return
    details.add_row(label, value if isinstance(value, Text) else Text(str(value)))
