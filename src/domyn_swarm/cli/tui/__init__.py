from typing import Iterable, Optional, Tuple

from rich.console import Console

from ...platform.protocols import ServingStatus
from ..tui.status import render_swarm_status
from ..tui.status_list import render_multi_status
from .list_view import render_swarm_list  # noqa: F401


def render_status(
    items: Iterable[Tuple[str, str, ServingStatus]] | Tuple[str, str, ServingStatus],
    *,
    console: Optional[Console] = None,
) -> None:
    """
    Convenience wrapper:
      - If you pass a single (name, backend, status) tuple, render full-width single view.
      - If you pass an iterable of tuples, render compact multi view.
    """
    if isinstance(items, tuple) and len(items) == 3:
        name, backend, st = items
        render_swarm_status(name, backend, st, console=console)  # type: ignore[arg-type]
        return

    items_list = list(items)  # type: ignore[arg-type]
    if len(items_list) == 1:
        name, backend, st = items_list[0]
        render_swarm_status(name, backend, st, console=console)
    else:
        render_multi_status(items_list, console=console)


__all__ = ["render_swarm_status", "render_multi_status", "render_status"]
