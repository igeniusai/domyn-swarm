#!/usr/bin/env python3
"""
Simple stdlib-only watchdog for a single vLLM replica.

Responsibilities:
- Spawn vllm serve (or any child command) and monitor it.
- Periodically:
  * Check HTTP readiness / liveness.
  * Optionally probe Ray cluster via CLI (Ray-aware mode).
  * Update a small SQLite table with status/heartbeat.
- Automatically restart the child on crash according to a restart policy.

Intended usage (from Slurm sbatch/srun):

  srun ... bash -lc '
    python -m domyn_swarm.runtime.watchdog \
      --db "$DSWARM_DB" \
      --swarm-id "$DSWARM_SWARM_ID" \
      --replica-id "$DSWARM_REPLICA_ID" \
      --node "$DSWARM_NODE" \
      --port "$DSWARM_PORT" \
      --agent-version "$DSWARM_AGENT_VERSION" \
      --restart-policy on-failure \
      --ray-enabled 0 \
      -- http child args... vllm serve ... \
  '

In Ray-aware mode, you pass a --ray-exec-prefix that prefixes Ray CLI calls and
(optionally) the vLLM child process, e.g.:

  --ray-enabled 1 \
  --ray-expected-tp 8 \
  --ray-timeout 120 \
  --ray-grace 10 \
  --ray-exec-prefix singularity exec instance://ray_head \
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import contextlib
from dataclasses import dataclass, field
import enum
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time
from urllib import error as urlerror, request

# ---------------------------------------------------------------------------
# Replica state model
# ---------------------------------------------------------------------------

MAX_REASON_LEN = 2048

REPLICA_STATUS_TABLE = "replica_status"


class ReplicaState(str, enum.Enum):
    STARTING = "starting"
    RUNNING = "running"
    UNHEALTHY = "unhealthy"
    EXITED = "exited"
    FAILED = "failed"
    RESTARTING = "restarting"


@dataclass
class WatchdogRayConfig:
    enabled: bool = False
    expected_tp: int | None = None  # None => don't gate on GPU capacity
    probe_timeout_s: float = 120.0
    status_grace_s: float = 10.0
    probe_interval_s: float = 10.0  # how often to check Ray once child is running


@dataclass
class WatchdogConfig:
    # HTTP health
    host: str = "127.0.0.1"
    port: int = 8000
    http_path: str = "/v1/models"
    http_timeout_s: float = 2.0
    probe_interval_s: float = 5.0
    unhealthy_http_failures: int = 3

    # Restart
    restart_policy: str = "on-failure"  # "always" | "on-failure" | "never"
    restart_backoff_s: float = 10.0
    max_restarts: int | None = None  # None => unlimited

    # Metadata
    agent_version: str = "unknown"

    # Ray-aware options
    ray: WatchdogRayConfig = field(default_factory=WatchdogRayConfig)

    def http_url(self) -> str:
        return f"http://{self.host}:{self.port}{self.http_path}"


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------


def open_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path.as_posix(), timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {REPLICA_STATUS_TABLE} (
          swarm_id      TEXT NOT NULL,
          replica_id    INTEGER NOT NULL,
          node          TEXT,
          port          INTEGER,
          pid           INTEGER,
          state         TEXT,
          http_ready    INTEGER,
          exit_code     INTEGER,
          exit_signal   INTEGER,
          fail_reason   TEXT,
          agent_version TEXT,
          last_seen     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          PRIMARY KEY (swarm_id, replica_id)
        );
        """
    )
    conn.commit()


def upsert_status(
    conn: sqlite3.Connection,
    *,
    swarm_id: str,
    replica_id: int,
    node: str | None,
    port: int | None,
    pid: int | None,
    state: ReplicaState,
    http_ready: bool,
    exit_code: int | None,
    exit_signal: int | None,
    fail_reason: str | None,
    agent_version: str,
) -> None:
    conn.execute(
        f"""
        INSERT INTO {REPLICA_STATUS_TABLE}
          (swarm_id, replica_id, node, port, pid, state, http_ready,
           exit_code, exit_signal, agent_version, last_seen)
        VALUES
          (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT (swarm_id, replica_id) DO UPDATE SET
          node          = excluded.node,
          port          = excluded.port,
          pid           = excluded.pid,
          state         = excluded.state,
          http_ready    = excluded.http_ready,
          exit_code     = excluded.exit_code,
          exit_signal   = excluded.exit_signal,
          fail_reason   = excluded.fail_reason,
          agent_version = excluded.agent_version,
          last_seen     = CURRENT_TIMESTAMP;
        """,
        (
            swarm_id,
            replica_id,
            node,
            port,
            pid,
            state.value,
            1 if http_ready else 0,
            exit_code,
            exit_signal,
            fail_reason,
            agent_version,
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Failure reason builder
# ---------------------------------------------------------------------------


def build_fail_reason(
    *,
    exit_code: int | None,
    exit_signal: int | None,
    log_path: str | None = None,
    restart_attempt: int | None = None,
    max_restarts: int | None = None,
) -> str:
    parts: list[str] = []

    # 1) Exit code / signal
    if exit_code is not None:
        parts.append(f"exit_code={exit_code}")
        if exit_code == 137:
            parts.append("(likely OOM / SIGKILL)")
    if exit_signal is not None:
        try:
            sig_name = signal.Signals(exit_signal).name
            parts.append(f"signal={sig_name}")
        except ValueError:
            parts.append(f"signal={exit_signal}")

    # 2) Restart context (optional)
    if restart_attempt is not None and max_restarts is not None:
        parts.append(f"restart_attempt={restart_attempt}/{max_restarts}")

    # 3) Log tail (optional & truncated)
    if log_path and os.path.exists(log_path):
        try:
            with open(log_path, "rb") as f:
                tail_bytes = f.read()[-4096:]
            tail = tail_bytes.decode(errors="replace")
            lines = tail.splitlines()[-10:]  # last 10 lines
            parts.append("log_tail:\n" + "\n".join(lines))
        except Exception:
            # If we fail to read logs, don't block the reason
            parts.append(f"log_tail: <unavailable: error reading {log_path}>")

    reason = " | ".join(parts).strip()
    if len(reason) > MAX_REASON_LEN:
        reason = reason[: MAX_REASON_LEN - 3] + "..."
    return reason or "unknown failure"


# ---------------------------------------------------------------------------
# HTTP & Ray probes
# ---------------------------------------------------------------------------


def _check_http(url: str, timeout: float) -> bool:
    try:
        with request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (urlerror.URLError, urlerror.HTTPError, TimeoutError, OSError):
        return False


def _run_cmd(
    argv: Sequence[str],
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _ray_cluster_ok(ray_prefix: Sequence[str]) -> bool:
    try:
        p = _run_cmd(
            [*ray_prefix, "ray", "status"],
            timeout=10.0,
        )
        return p.returncode == 0
    except Exception:
        return False


def _ray_gpu_capacity_ok(
    ray_prefix: Sequence[str],
    expected_tp: int | None,
) -> bool:
    if expected_tp is None or expected_tp <= 0:
        return True

    try:
        p = _run_cmd(
            [*ray_prefix, "ray", "list", "nodes", "--format=json"],
            timeout=10.0,
        )
        if p.returncode != 0:
            return False
        data = json.loads(p.stdout or "[]")
        total_gpu = 0.0
        for n in data:
            if n.get("state") != "ALIVE":
                continue
            resources = n.get("resources", {}) or {}
            total_gpu += float(resources.get("GPU", 0))
        return total_gpu >= float(expected_tp)
    except Exception:
        return False


def _ray_probe_once(ray_cfg: WatchdogRayConfig, ray_prefix: Sequence[str]) -> bool:
    if not ray_cfg.enabled:
        return True
    alive = _ray_cluster_ok(ray_prefix)
    capacity = _ray_gpu_capacity_ok(ray_prefix, ray_cfg.expected_tp)
    return alive and capacity


# ---------------------------------------------------------------------------
# Main watchdog logic
# ---------------------------------------------------------------------------


@dataclass
class ReplicaMeta:
    swarm_id: str
    replica_id: int
    node: str
    port: int


def _mark_state(
    conn: sqlite3.Connection,
    meta: ReplicaMeta,
    cfg: WatchdogConfig,
    *,
    state: ReplicaState,
    http_ready: bool,
    pid: int | None,
    exit_code: int | None,
    exit_signal: int | None,
    fail_reason: str | None,
) -> None:
    upsert_status(
        conn,
        swarm_id=meta.swarm_id,
        replica_id=meta.replica_id,
        node=meta.node,
        port=meta.port,
        pid=pid,
        state=state,
        http_ready=http_ready,
        exit_code=exit_code,
        exit_signal=exit_signal,
        fail_reason=fail_reason,
        agent_version=cfg.agent_version,
    )


def _spawn_child_and_mark_running(
    conn: sqlite3.Connection,
    meta: ReplicaMeta,
    cfg: WatchdogConfig,
    child_argv: Sequence[str],
) -> tuple[subprocess.Popen, int]:
    # Mark STARTING
    _mark_state(
        conn,
        meta,
        cfg,
        state=ReplicaState.STARTING,
        http_ready=False,
        pid=None,
        exit_code=None,
        exit_signal=None,
        fail_reason=None,
    )

    # Spawn child
    child = subprocess.Popen(
        list(child_argv),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    pid = child.pid or 0

    # Mark RUNNING
    _mark_state(
        conn,
        meta,
        cfg,
        state=ReplicaState.RUNNING,
        http_ready=False,
        pid=pid,
        exit_code=None,
        exit_signal=None,
        fail_reason=None,
    )
    return child, pid


def _should_restart(exit_code: int, cfg: WatchdogConfig, restart_count: int) -> bool:
    if cfg.restart_policy == "never":
        return False
    if cfg.restart_policy == "on-failure" and exit_code == 0:
        return False
    return not (cfg.max_restarts is not None and restart_count >= cfg.max_restarts)


def _probe_and_update(
    conn: sqlite3.Connection,
    meta: ReplicaMeta,
    cfg: WatchdogConfig,
    pid: int,
    http_failures: int,
    http_ok_since: float | None,
    ray_ok_since: float | None,
    last_ray_probe: float,
    ray_prefix: Sequence[str],
) -> tuple[int, float | None, float | None, float]:
    now = time.time()
    url = cfg.http_url()

    # HTTP probe
    http_ok = _check_http(url, cfg.http_timeout_s)
    if http_ok:
        http_failures = 0
        http_ok_since = http_ok_since or now
    else:
        http_failures += 1
        http_ok_since = None

    # Ray probe (optional)
    ray_ready = True
    if cfg.ray.enabled:
        if now - last_ray_probe >= cfg.ray.probe_interval_s:
            ray_single_ok = _ray_probe_once(cfg.ray, ray_prefix)
            last_ray_probe = now
            ray_ok_since = now if ray_single_ok else None

        ray_ready = ray_ok_since is not None and now - ray_ok_since >= cfg.ray.status_grace_s

    http_ready_flag = bool(http_ok_since) and (not cfg.ray.enabled or ray_ready)

    # Running vs Unhealthy
    state = (
        ReplicaState.UNHEALTHY
        if http_failures >= cfg.unhealthy_http_failures
        else ReplicaState.RUNNING
    )

    _mark_state(
        conn,
        meta,
        cfg,
        state=state,
        http_ready=http_ready_flag,
        pid=pid,
        exit_code=None,
        exit_signal=None,
        fail_reason=None,
    )

    return http_failures, http_ok_since, ray_ok_since, last_ray_probe


def _monitor_child_loop(
    conn: sqlite3.Connection,
    meta: ReplicaMeta,
    cfg: WatchdogConfig,
    child: subprocess.Popen,
    pid: int,
    stop_flag: dict,
    ray_prefix: Sequence[str],
    restart_count: int,
    log_dir: Path,
) -> tuple[int, bool, str | None]:
    """Monitor a single child process until it exits or we get a stop signal.

    Returns (exit_code, should_restart).
    """
    http_failures = 0
    http_ok_since: float | None = None
    ray_ok_since: float | None = None
    last_ray_probe = 0.0
    fail_reason: str | None = None

    while True:
        if stop_flag.get("stop"):
            # Graceful shutdown
            if child.poll() is None:
                with contextlib.suppress(Exception):
                    child.terminate()
            child.wait(timeout=10)
            _mark_state(
                conn,
                meta,
                cfg,
                state=ReplicaState.EXITED,
                http_ready=False,
                pid=pid,
                exit_code=child.returncode,
                exit_signal=None,
                fail_reason=None,
            )
            return 0, False, None

        ret = child.poll()
        if ret is not None:
            # Child exited
            state = ReplicaState.EXITED if ret == 0 else ReplicaState.FAILED

            if state == ReplicaState.FAILED:
                log_path = log_dir / "vllm.log"
                fail_reason = build_fail_reason(
                    exit_code=ret,
                    log_path=log_path.as_posix() if log_path.exists() else None,
                    exit_signal=None,
                    restart_attempt=restart_count + 1,
                    max_restarts=cfg.max_restarts,
                )

            _mark_state(
                conn,
                meta,
                cfg,
                state=state,
                http_ready=False,
                pid=pid,
                exit_code=ret,
                exit_signal=None,
                fail_reason=fail_reason,
            )
            should_restart = _should_restart(ret, cfg, restart_count)
            return ret, should_restart, fail_reason

        # Still running → probe and update status
        http_failures, http_ok_since, ray_ok_since, last_ray_probe = _probe_and_update(
            conn,
            meta,
            cfg,
            pid,
            http_failures,
            http_ok_since,
            ray_ok_since,
            last_ray_probe,
            ray_prefix,
        )

        time.sleep(cfg.probe_interval_s)


def run_watchdog(
    db_path: Path,
    swarm_id: str,
    replica_id: int,
    node: str,
    port: int,
    cfg: WatchdogConfig,
    child_argv: Sequence[str],
    log_dir: Path,
    ray_exec_prefix: Sequence[str] | None = None,
) -> int:
    """
    Orchestrate spawn/monitor/restart for a single replica.

    - Spawns child
    - Monitors it until exit or stop signal
    - Decides whether to restart based on restart_policy/max_restarts
    """
    if not child_argv:
        print("watchdog: no child command provided", file=sys.stderr)
        return 1

    cfg.port = port

    db_conn = open_db(db_path)
    ray_prefix: Sequence[str] = ray_exec_prefix or []

    meta = ReplicaMeta(
        swarm_id=swarm_id,
        replica_id=replica_id,
        node=node,
        port=port,
    )

    stop_flag = {"stop": False}

    def _handle_sigterm(signum, frame):
        stop_flag["stop"] = True

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    restart_count = 0

    while True:
        # 1) Spawn child and mark STARTING/RUNNING
        child, pid = _spawn_child_and_mark_running(db_conn, meta, cfg, child_argv)

        # 2) Monitor until it exits or we get a stop signal
        exit_code, should_restart, fail_reason = _monitor_child_loop(
            db_conn,
            meta,
            cfg,
            child,
            pid,
            stop_flag,
            ray_prefix,
            restart_count,
            log_dir,
        )

        if not should_restart:
            return exit_code

        restart_count += 1

        # 3) Mark RESTARTING and backoff before re-spawn
        _mark_state(
            db_conn,
            meta,
            cfg,
            state=ReplicaState.RESTARTING,
            http_ready=False,
            pid=None,
            exit_code=exit_code,
            exit_signal=None,
            fail_reason=fail_reason,
        )
        time.sleep(cfg.restart_backoff_s)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Domyn-Swarm watchdog (spawn & monitor a single vLLM replica)."
    )
    parser.add_argument("--db", required=True, help="Path to per-swarm SQLite DB.")
    parser.add_argument("--swarm-id", required=True, help="Swarm identifier.")
    parser.add_argument("--replica-id", type=int, required=True, help="Replica index.")
    parser.add_argument("--node", required=True, help="Node hostname for diagnostics.")
    parser.add_argument("--port", type=int, required=True, help="HTTP port for vLLM.")
    parser.add_argument(
        "--agent-version",
        default=os.environ.get("DSWARM_AGENT_VERSION", "unknown"),
        help="Watchdog/agent version string.",
    )

    # HTTP / restart tuning (basic subset; can be overridden from YAML/env in your launcher)
    parser.add_argument(
        "--http-path",
        type=str,
        default="/health",
        help="HTTP health check path (default: /health).",
    )
    parser.add_argument(
        "--probe-interval",
        type=float,
        default=5.0,
        help="Probe interval in seconds (default: 5).",
    )
    parser.add_argument(
        "--http-timeout",
        type=float,
        default=2.0,
        help="HTTP timeout in seconds (default: 2).",
    )
    parser.add_argument(
        "--restart-policy",
        choices=["always", "on-failure", "never"],
        default="on-failure",
        help="Restart policy for child process (default: on-failure).",
    )
    parser.add_argument(
        "--restart-backoff",
        type=float,
        default=10.0,
        help="Seconds to sleep before restarting the child (default: 10).",
    )
    parser.add_argument(
        "--max-restarts",
        type=int,
        default=None,
        help="Maximum number of restarts (default: unlimited).",
    )
    parser.add_argument(
        "--unhealthy-http-failures",
        type=int,
        default=3,
        help="Consecutive HTTP failures before marking replica UNHEALTHY.",
    )

    # Ray-aware flags
    parser.add_argument(
        "--ray-enabled",
        type=int,
        default=0,
        help="Enable Ray-aware checks (0/1). Default: 0 (disabled).",
    )
    parser.add_argument(
        "--ray-expected-tp",
        type=int,
        default=0,
        help="Expected tensor-parallel (GPU) capacity. 0 => ignore capacity.",
    )
    parser.add_argument(
        "--ray-timeout",
        type=float,
        default=120.0,
        help="Max seconds to wait for Ray to become healthy (used via grace window).",
    )
    parser.add_argument(
        "--ray-grace",
        type=float,
        default=10.0,
        help="Seconds Ray must stay healthy before considered ready (grace window).",
    )
    parser.add_argument(
        "--ray-probe-interval",
        type=float,
        default=10.0,
        help="How often to probe Ray once child is running.",
    )
    parser.add_argument(
        "--ray-exec-prefix",
        nargs="*",
        default=None,
        help=(
            "Command prefix for Ray CLI (e.g. 'singularity exec instance://ray_head'). "
            "If omitted, Ray CLI is run directly (inside container)."
        ),
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        required=True,
        help="Directory where vLLM logs are stored.",
    )

    parser.add_argument(
        "child",
        nargs=argparse.REMAINDER,
        help="Child command to run (must appear after '--').",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    # Child argv: strip leading '--' if present because of argparse semantics
    child_argv = list(args.child)
    if child_argv and child_argv[0] == "--":
        child_argv = child_argv[1:]

    if not child_argv:
        print("watchdog: no child command provided after '--'", file=sys.stderr)
        return 1

    cfg = WatchdogConfig(
        host="127.0.0.1",  # typically localhost inside the replica node
        port=args.port,
        http_path=args.http_path,
        http_timeout_s=args.http_timeout,
        probe_interval_s=args.probe_interval,
        unhealthy_http_failures=args.unhealthy_http_failures,
        restart_policy=args.restart_policy,
        restart_backoff_s=args.restart_backoff,
        max_restarts=args.max_restarts,
        agent_version=args.agent_version,
        ray=WatchdogRayConfig(
            enabled=bool(args.ray_enabled),
            expected_tp=args.ray_expected_tp or None,
            probe_timeout_s=args.ray_timeout,
            status_grace_s=args.ray_grace,
            probe_interval_s=args.ray_probe_interval,
        ),
    )

    db_path = Path(args.db)
    ray_prefix: Sequence[str] | None = args.ray_exec_prefix or None

    return run_watchdog(
        db_path=db_path,
        swarm_id=args.swarm_id,
        replica_id=args.replica_id,
        node=args.node,
        port=args.port,
        cfg=cfg,
        child_argv=child_argv,
        ray_exec_prefix=ray_prefix,
        log_dir=Path(args.log_dir),
    )


if __name__ == "__main__":
    raise SystemExit(main())
