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

"""``domyn-swarm monitor`` — launch grafatui against a swarm's Prometheus.

Lean by design: resolve the proxied Prometheus URL from persisted swarm state and
exec grafatui (an optional external tool) with the bundled vLLM dashboard, or a
custom dashboard via ``--dashboard``. If grafatui is absent, print the URL and an
install hint so the user can use Grafana or run it manually.
"""

from __future__ import annotations

from importlib import resources
import os
from pathlib import Path
import shutil
from typing import Annotated, Any

import typer


def build_prometheus_url(swarm) -> str:
    """Return the proxied Prometheus URL for a loaded swarm.

    Args:
        swarm: A loaded swarm object exposing ``endpoint`` and
            ``cfg.backend.endpoint.monitoring.route_prefix``.

    Returns:
        ``<endpoint><route_prefix>`` with the endpoint's trailing slash removed.
    """
    base = swarm.endpoint.rstrip("/")
    prefix = swarm.cfg.backend.endpoint.monitoring.route_prefix
    if not prefix.startswith("/"):
        prefix = "/" + prefix
    return f"{base}{prefix}"


def resolve_grafatui_argv(
    url: str,
    *,
    dashboard: Path | None,
    extra: list[str],
    variables: dict[str, str] | None = None,
) -> list[str]:
    """Build the grafatui argument vector.

    Args:
        url: Prometheus URL to pass via ``--prometheus-url``.
        dashboard: Optional Grafana dashboard JSON to load via ``--grafana-json``.
        extra: Additional passthrough arguments (e.g. ``["--range", "1h"]``).
        variables: Dashboard template variables to pass as repeated ``--var KEY=VALUE``.

    Returns:
        The full argv beginning with ``grafatui``.
    """
    argv = ["grafatui", "--prometheus-url", url]
    if dashboard is not None:
        argv += ["--grafana-json", str(dashboard)]
    for key, value in (variables or {}).items():
        argv += ["--var", f"{key}={value}"]
    argv += extra
    return argv


def _parse_var_overrides(pairs: list[str]) -> dict[str, str]:
    """Parse ``KEY=VALUE`` strings into a dict, raising on malformed entries."""
    out: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise typer.BadParameter(f"--var must be KEY=VALUE, got: {pair!r}")
        out[key] = value
    return out


def _bundled_dashboard() -> Path | None:
    """Return a filesystem path to the bundled vLLM dashboard JSON, or None."""
    try:
        ref = resources.files("domyn_swarm.data.dashboards").joinpath("vllm.json")
        with resources.as_file(ref) as p:
            return Path(p)
    except (ModuleNotFoundError, FileNotFoundError):
        return None


def monitor(
    name: Annotated[str, typer.Argument(help="Swarm name to monitor.")],
    dashboard: Annotated[
        Path | None,
        typer.Option(
            "--dashboard",
            "-d",
            help="Custom Grafana dashboard JSON to load. Defaults to the bundled vLLM dashboard.",
        ),
    ] = None,
    prometheus_url: Annotated[
        str | None, typer.Option("--prometheus-url", help="Override the resolved Prometheus URL.")
    ] = None,
    range_: Annotated[
        str | None, typer.Option("--range", help="grafatui time range, e.g. 1h.")
    ] = None,
    step: Annotated[
        str | None, typer.Option("--step", help="grafatui query step, e.g. 15s.")
    ] = None,
    var: Annotated[
        list[str] | None,
        typer.Option(
            "--var",
            help="Override a dashboard variable as KEY=VALUE (repeatable). "
            "Auto-filled defaults: vllm_job, replicas.",
        ),
    ] = None,
) -> None:
    """Launch grafatui pointed at the swarm's Prometheus endpoint."""
    from domyn_swarm.core.state.state_manager import SwarmStateManager

    swarm: Any = SwarmStateManager.load(deployment_name=name)
    mon = getattr(swarm.cfg.backend.endpoint, "monitoring", None)
    if not getattr(mon, "enabled", False):
        typer.echo(
            f"Monitoring is not enabled for swarm '{name}'. "
            "Set backend.endpoint.monitoring.enabled: true and redeploy.",
            err=True,
        )
        raise typer.Exit(code=1)

    url = prometheus_url or build_prometheus_url(swarm)

    # Dashboard variables: defaults derived from the swarm config, overridable via --var.
    # We intentionally do NOT auto-fill `swarm`/`model`: a swarm's Prometheus scrapes
    # only that one swarm (so those filters are redundant), and vLLM labels `model_name`
    # with the full resolved model path — not cfg.model — so it would match nothing.
    # Users can still pass either explicitly via --var for custom dashboards.
    variables: dict[str, str] = {"vllm_job": "vllm"}
    cfg = swarm.cfg
    if getattr(cfg, "replicas", None) is not None:
        variables["replicas"] = str(cfg.replicas)
    variables.update(_parse_var_overrides(var or []))

    extra: list[str] = []
    if range_:
        extra += ["--range", range_]
    if step:
        extra += ["--step", step]

    if shutil.which("grafatui") is None:
        typer.echo(
            "grafatui not found on PATH. Prometheus is available at:\n"
            f"  {url}\n"
            "Install grafatui (`cargo install grafatui` or a GitHub release binary), "
            "or point Grafana at that URL.",
            err=True,
        )
        raise typer.Exit(code=127)

    if dashboard is not None:
        if not dashboard.is_file():
            typer.echo(f"Dashboard file not found: {dashboard}", err=True)
            raise typer.Exit(code=2)
    else:
        dashboard = _bundled_dashboard()
    argv = resolve_grafatui_argv(url, dashboard=dashboard, extra=extra, variables=variables)
    os.execvp(argv[0], argv)
