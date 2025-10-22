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

import json
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Optional

import typer
from rich.console import Console

from domyn_swarm.helpers.logger import setup_logger


# DTO for the view layer
@dataclass
class SwarmSummary:
    name: str
    backend: str  # "slurm" | "lepton" | ...
    phase: str  # str(ServingPhase) or best-effort string
    url: Optional[str] = None
    http: Optional[int | str] = None
    extra: dict[str, Any] | None = None


logger = setup_logger(__name__)
swarm_app = typer.Typer(help="List existing swarms with a compact status view.")


def _iter_summaries(*, probe: bool) -> Iterable[SwarmSummary]:
    """
    Adapter from your state layer → SwarmSummary DTOs used by the renderer.
    Tries to be compatible with your current state manager.
    """
    # Lazy imports to avoid CLI import cost for non-list commands
    from domyn_swarm.core.state import SwarmStateManager
    from domyn_swarm.core.swarm import DomynLLMSwarm
    from domyn_swarm.platform.protocols import ServingPhase

    records = SwarmStateManager.list_all()

    for rec in records:
        name: str = rec.get("name", "unnamed-swarm")
        backend = rec.get("platform", "unknown").lower()

        url = rec.get("endpoint", "")
        phase = ServingPhase.UNKNOWN.value
        http = None
        extra: dict[str, Any] | None = None

        if name and probe:
            try:
                # Load a live swarm and ask the serving backend for status
                swarm = DomynLLMSwarm.from_state(deployment_name=name)
                # Prefer serving.status(handle) if present
                st = swarm._deployment.serving.status(swarm.serving_handle)  # type: ignore[attr-defined]
                phase = st.phase.value
                if phase == "UNKNOWN":
                    continue  # skip dead/unknown swarms

                # Prefer URL from live status if available
                url = url or st.url
                if st.detail:
                    http = st.detail.get("http", None)
                    extra = {}
                    for k in ("rep", "lb", "raw_state"):
                        if k in st.detail:
                            extra[k] = st.detail[k]
            except Exception as e:
                logger.debug(f"Status probe failed for {name}: {e}")

        yield SwarmSummary(
            name=name, backend=backend, phase=phase, url=url, http=http, extra=extra
        )


@swarm_app.command("list")
def list_swarms(
    probe: bool = typer.Option(
        True,
        "--probe/--no-probe",
        help="Probe live status (HTTP/LB/etc.). Disable for faster listing.",
    ),
):
    """
    List all known swarms in a compact table (name, backend, phase, endpoint, notes).
    """
    from domyn_swarm.cli.tui.list_view import render_swarm_list

    console = Console()
    rows = list(_iter_summaries(probe=probe))
    if not rows:
        console.print("[yellow]No swarms found.[/]")
        raise typer.Exit(0)

    render_swarm_list(rows, console=console)


@swarm_app.command("describe")
def describe_swarm(
    name: str = typer.Argument(..., help="Name of the swarm to describe."),
    output: Literal["table", "yaml", "json"] = typer.Option(
        "table",
        "--output",
        "-o",
        case_sensitive=False,
        help="Output format: table (default), yaml, or json.",
    ),
):
    """
    Show a detailed, static description of a swarm from the local state.
    No live status probing (use `domyn-swarm status` for that).
    """
    import yaml

    from domyn_swarm.core.state import SwarmStateManager

    # Load a single record from state
    swarm = SwarmStateManager.load(deployment_name=name)

    backend = swarm._platform.lower()
    endpoint = swarm.endpoint or ""
    cfg = swarm.cfg
    data = cfg.model_dump(mode="json", by_alias=True, exclude_none=True)

    if output.lower() == "table":
        # Rich TUI view
        from domyn_swarm.cli.tui.describe_view import render_swarm_description

        console = Console()
        render_swarm_description(
            name=name,
            backend=backend,
            cfg=data,
            endpoint=endpoint,
            console=console,
        )
        return

    # Structured payload for non-TUI outputs
    payload = {
        "name": name,
        "backend": backend,
        "endpoint": endpoint,
        "config": data,
    }

    if output.lower() == "yaml":
        typer.echo(yaml.safe_dump(payload, sort_keys=False))
    elif output.lower() == "json":
        typer.echo(json.dumps(payload, indent=2))
    else:
        # Fallback to table if something odd sneaks in
        from domyn_swarm.cli.tui.describe_view import render_swarm_description

        console = Console()
        render_swarm_description(
            name=name,
            backend=backend,
            cfg=data,
            endpoint=endpoint,
            console=console,
        )
