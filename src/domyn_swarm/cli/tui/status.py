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

from rich.box import HEAVY
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from domyn_swarm.cli.tui.components import (
    _add_if,
    _color_state,
    _fmt_http,
    _phase_badge,
)
from domyn_swarm.cli.tui.tables import _kv_table
from domyn_swarm.platform.protocols import ServingStatus


# ---- Styling helpers --------------------------------------------------------
def render_swarm_status(
    name: str,
    backend: str,
    st: ServingStatus,
    *,
    console: Console | None = None,
) -> None:
    """Renders a comprehensive status view for a single swarm deployment.

    Displays a full-width status panel with swarm information including name, backend,
    phase, endpoint URL, and diagnostic details. The output is optimized for terminal
    display with styled text, tables, and panels using the Rich library.

    Args:
        name: The name/identifier of the swarm to display.
        backend: The deployment backend (e.g., 'slurm', 'lepton'). Will be displayed
            in uppercase.
        st: The serving status object containing phase, URL, and detailed diagnostic
            information.
        console: Optional Rich Console instance for output. If None, creates a new
            Console instance.

    Note:
        The diagnostic section adapts to different backends:
        - For Slurm: Shows replica ('rep') and load balancer ('lb') states
        - For Lepton: Shows raw platform state ('raw_state')
        - HTTP status is displayed for all backends when available
        - Additional diagnostic keys are shown in an 'Extra' section

    Example:
        >>> status = ServingStatus(phase="running", url="http://localhost:8000")
        >>> render_swarm_status("my-swarm", "slurm", status)
    """
    console = console or Console()

    banner = Text.assemble(
        ("SWARM ", "bold dim"),
        (name, "bold bright_cyan"),
        ("  •  ", "dim"),
        (backend.upper(), "bold magenta"),
    )

    info = _kv_table()
    info.add_row("Phase", _phase_badge(st.phase))
    url_txt = Text(st.url or "—")
    if st.url:
        url_txt.stylize(f"link {st.url}")
    info.add_row("Endpoint", url_txt)

    # Diagnostics section (backend-agnostic, but friendly to Slurm/Lepton)
    diag = _kv_table()
    _add_if(diag, "HTTP", _fmt_http(st.detail))

    if st.detail:
        # Slurm-style signals
        rep = st.detail.get("rep")
        lb = st.detail.get("lb")
        if rep is not None or lb is not None:
            rep_txt = _color_state(rep)
            lb_txt = _color_state(lb)
            diag.add_row("Replica", rep_txt)
            diag.add_row("LB", lb_txt)

        # Lepton raw platform state
        raw = st.detail.get("raw_state")
        if raw is not None:
            diag.add_row("Platform", _color_state(raw))

        # Any remaining keys → Extras
        extras = {k: v for k, v in st.detail.items() if k not in {"http", "rep", "lb", "raw_state"}}
        if extras:
            extras_kv = ", ".join(f"{k}={v}" for k, v in extras.items())
            diag.add_row("Extra", Text(extras_kv, overflow="fold"))

    body = Group(
        Rule(banner, style="cyan"),
        Text(),  # spacer
        info,
        Text(),  # spacer
        Panel(diag, title="Diagnostics", border_style="dim", padding=(1, 2), box=HEAVY)
        if len(diag.rows) > 0
        else Text(""),
    )

    panel = Panel(
        body,
        title=f"[b cyan]{name}[/] — [magenta]{backend.upper()}[/]",
        subtitle=(st.url or ""),
        border_style="cyan",
        padding=(1, 2),
        box=HEAVY,
        expand=True,
    )
    console.print(panel)
