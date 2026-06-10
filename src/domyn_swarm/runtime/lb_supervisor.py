#!/usr/bin/env python3
"""Domyn-Swarm load-balancer supervisor.

Keeps nginx's dynamic upstreams (and, when monitoring is enabled, the Prometheus
``targets.json``) in sync with the ``replica-*.head`` files that each vLLM replica
writes into the serving directory. Runs inside the same Python image as the
collector and is launched by the LB sbatch job (see ``templates/lb.sh.j2``).

Generation-only: the supervisor writes ``00-upstreams.conf`` and, when
``--emit-targets`` is set, ``targets.json``. It does **not** reload nginx.
nginx reload is performed host-side by ``lb.sh.j2`` via ``inotifywait`` (or
equivalent polling). Cross-container ``SIGHUP`` is not possible on this platform
because ``singularity instance start`` uses a private PID namespace, so a process
inside the vLLM container cannot signal the nginx master process running in a
separate Singularity instance.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import sys
import time


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


def write_if_changed(target: Path, content: str) -> bool:
    """Atomically write ``content`` to ``target`` only if it differs.

    Writes to a sibling temp file and ``os.replace`` so readers never observe a
    partial file.

    Args:
        target: Destination path.
        content: Text to write.

    Returns:
        True if the file was created or its content changed; False if the
        existing content already matched.
    """
    target = Path(target)
    if target.exists() and target.read_text() == content:
        return False
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(content)
    os.replace(tmp, target)
    return True


UPSTREAMS_FILENAME = "00-upstreams.conf"
TARGETS_FILENAME = "targets.json"


def render_targets(serving_dir: Path) -> str:
    """Render Prometheus file_sd targets JSON for the current replicas.

    Args:
        serving_dir: Directory containing ``replica-*.head`` files.

    Returns:
        A JSON string with a single file_sd entry labelling all replica
        ``host:port`` targets with ``job=vllm``.
    """
    addrs = read_head_files(serving_dir)
    return json.dumps([{"targets": addrs, "labels": {"job": "vllm"}}]) + "\n"


@dataclass
class SupervisorOptions:
    """Runtime options for the LB supervisor reconcile loop.

    Attributes:
        serving_dir: Directory holding ``replica-*.head`` and generated conf.d files.
        ray_enabled: Whether to emit Ray upstreams.
        ray_dashboard_port: Ray dashboard port.
        ray_port: Ray client port.
        emit_targets: Whether to also write the Prometheus file_sd targets.json.
    """

    serving_dir: Path
    ray_enabled: bool
    ray_dashboard_port: int
    ray_port: int
    emit_targets: bool = False


def reconcile_once(opts: SupervisorOptions) -> bool:
    """Regenerate the upstreams file and targets from the current head files.

    Writes ``00-upstreams.conf`` (and ``targets.json`` when
    ``opts.emit_targets`` is set) to ``opts.serving_dir``. Does **not** reload
    nginx — reload is performed host-side by ``lb.sh.j2``.

    Args:
        opts: Supervisor options.

    Returns:
        True if the upstreams file changed, else False.
    """
    conf = render_upstreams(
        opts.serving_dir,
        ray_enabled=opts.ray_enabled,
        ray_dashboard_port=opts.ray_dashboard_port,
        ray_port=opts.ray_port,
    )
    changed = write_if_changed(opts.serving_dir / UPSTREAMS_FILENAME, conf)
    if opts.emit_targets:
        write_if_changed(opts.serving_dir / TARGETS_FILENAME, render_targets(opts.serving_dir))
    return changed


def run_loop(opts: SupervisorOptions, interval: int) -> int:
    """Continuously reconcile upstreams with head files until terminated.

    Args:
        opts: Supervisor options.
        interval: Seconds to sleep between reconcile passes.

    Returns:
        Process exit code (0 on clean shutdown via SIGTERM/SIGINT).
    """
    stop = {"flag": False}

    def _handle(signum, frame):
        print(f"lb_supervisor: received signal {signum}, stopping", file=sys.stderr)
        stop["flag"] = True

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    print(
        f"lb_supervisor: watching {opts.serving_dir} (interval={interval}s, "
        f"ray_enabled={opts.ray_enabled})",
        file=sys.stderr,
    )
    while not stop["flag"]:
        try:
            reconcile_once(opts)
        except Exception as e:  # never let a transient error kill the loop
            print(f"lb_supervisor: reconcile error: {e!r}", file=sys.stderr)
        for _ in range(interval):
            if stop["flag"]:
                break
            time.sleep(1)
    return 0


def parse_args(argv: list[str] | None = None) -> tuple[SupervisorOptions, bool, int]:
    """Parse CLI arguments into supervisor options.

    Args:
        argv: Argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        A tuple of (options, run_once, interval_seconds).
    """
    p = argparse.ArgumentParser(description="Domyn-Swarm LB supervisor.")
    p.add_argument("--serving-dir", required=True)
    p.add_argument("--ray-enabled", action="store_true")
    p.add_argument("--ray-dashboard-port", type=int, default=8265)
    p.add_argument("--ray-port", type=int, default=6379)
    p.add_argument("--interval", type=int, default=5)
    p.add_argument("--once", action="store_true")
    p.add_argument("--emit-targets", action="store_true")
    a = p.parse_args(argv)
    opts = SupervisorOptions(
        serving_dir=Path(a.serving_dir),
        ray_enabled=a.ray_enabled,
        ray_dashboard_port=a.ray_dashboard_port,
        ray_port=a.ray_port,
        emit_targets=a.emit_targets,
    )
    return opts, a.once, a.interval


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the LB supervisor."""
    opts, once, interval = parse_args(argv)
    if once:
        reconcile_once(opts)
        return 0
    return run_loop(opts, interval)


if __name__ == "__main__":
    raise SystemExit(main())
