from typing import Iterable, Optional, Tuple

from rich.box import HEAVY
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from domyn_swarm.cli.tui.components import _color_state, _fmt_http, _phase_badge
from domyn_swarm.cli.tui.tables import _kv_table
from domyn_swarm.platform.protocols import ServingStatus


def render_multi_status(
    items: Iterable[Tuple[str, str, ServingStatus]],
    *,
    console: Optional[Console] = None,
) -> None:
    """Renders compact status panels for multiple swarms in a columnar layout.

    This function displays status information for multiple swarms in a compact,
    side-by-side panel format. Each panel shows key status information including
    phase, endpoint URL, HTTP details, and platform-specific state information.
    For single swarm displays, consider using render_swarm_status instead.

    Args:
        items: An iterable of tuples containing (name, backend, status) for each
            swarm to display. Each tuple contains:
            - name (str): The display name of the swarm
            - backend (str): The backend type/provider name
            - status (ServingStatus): The current serving status with details
        console: Optional Rich Console instance for output. If None, creates a
            new Console instance.

    Returns:
        None: Outputs directly to the console.

    Note:
        The function creates expandable, equal-width columns with cyan borders
        and heavy box styling. Status information includes phase badges,
        clickable endpoint URLs, HTTP status, replica states, load balancer
        endpoints, and raw platform state when available.
    """
    console = console or Console()
    panels = []
    for name, backend, st in items:
        tbl = _kv_table()
        tbl.add_row("Phase", _phase_badge(st.phase))
        url_txt = Text(st.url or "—")
        if st.url:
            url_txt.stylize(f"link {st.url}")
        tbl.add_row("Endpoint", url_txt)

        http = _fmt_http(st.detail)
        if http:
            tbl.add_row("HTTP", http)
        if st.detail:
            rep = st.detail.get("rep")
            endpoint = st.detail.get("lb")
            raw = st.detail.get("raw_state")
            if rep is not None:
                tbl.add_row("Replica", _color_state(rep))
            if endpoint is not None:
                tbl.add_row("Endpoint", _color_state(endpoint))
            if raw is not None:
                tbl.add_row("Platform", _color_state(raw))

        panels.append(
            Panel(
                tbl,
                title=f"[b]{name}[/] • {backend.upper()}",
                border_style="cyan",
                padding=(1, 2),
                box=HEAVY,
            )
        )
    console.print(Columns(panels, expand=True, equal=True, padding=1))
