#!/usr/bin/env python3
"""Domyn-Swarm load-balancer supervisor.

Keeps nginx's dynamic upstreams (and, when monitoring is enabled, the Prometheus
``targets.json``) in sync with the ``replica-*.head`` files that each vLLM replica
writes into the serving directory. Runs inside the same Python image as the
collector and is launched by the LB sbatch job (see ``templates/lb.sh.j2``).

Reload mechanism: the supervisor sends ``SIGHUP`` to the nginx master process
(PID read from ``<serving_dir>/../run/nginx.pid``). Singularity instances share
the host PID namespace by default, so this reaches nginx running in a separate
instance. See Phase 0 spike in the implementation plan.
"""

from __future__ import annotations

from pathlib import Path


def read_head_files(serving_dir: Path) -> list[str]:
    """Return replica ``host:port`` addresses, ordered by replica id.

    Args:
        serving_dir: Directory containing ``replica-<id>.head`` files, each
            holding a single ``host:port`` line.

    Returns:
        Addresses sorted ascending by the integer replica id encoded in the
        filename. Empty if no head files exist yet.
    """
    serving_dir = Path(serving_dir)
    entries: list[tuple[int, str]] = []
    for f in serving_dir.glob("replica-*.head"):
        try:
            replica_id = int(f.stem.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        addr = f.read_text().strip()
        if addr and ":" in addr:
            entries.append((replica_id, addr))
    entries.sort(key=lambda e: e[0])
    return [addr for _, addr in entries]


def render_upstreams(
    serving_dir: Path,
    *,
    ray_enabled: bool,
    ray_dashboard_port: int,
    ray_port: int,
) -> str:
    """Render the nginx ``upstream`` blocks for the current replicas.

    Reproduces the upstream generation previously embedded in ``lb.sh.j2``.

    Args:
        serving_dir: Directory containing ``replica-*.head`` files.
        ray_enabled: Whether to also emit the Ray dashboard/control upstreams
            (host taken from the lowest-id replica with a valid address
            (typically replica 0)).
        ray_dashboard_port: Ray dashboard port for the ``ray`` upstream.
        ray_port: Ray client port for the ``ray_control`` upstream.

    Returns:
        nginx configuration text defining the ``llm`` upstream (and Ray
        upstreams when enabled).
    """
    addrs = read_head_files(serving_dir)
    lines = ["upstream llm {", "  least_conn;"]
    lines.extend(f"  server {addr} max_fails=2 fail_timeout=10s;" for addr in addrs)
    lines.append("}")

    if ray_enabled and addrs:
        ray_host = addrs[0].split(":", 1)[0]
        lines += [
            "upstream ray {",
            f"  server {ray_host}:{ray_dashboard_port};",
            "  keepalive 8;",
            "}",
            "upstream ray_control {",
            f"  server {ray_host}:{ray_port};",
            "  keepalive 8;",
            "}",
        ]
    return "\n".join(lines) + "\n"
