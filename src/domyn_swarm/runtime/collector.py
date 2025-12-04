#!/usr/bin/env python3
"""
Domyn-Swarm watchdog collector (single writer).

- Listens on a TCP socket for JSON messages from watchdogs.
- Each message describes the status of one replica.
- Collector is the *only* writer to watchdog.db for a given swarm.

Example (in LB job):

  python3 /opt/watchdog_collector.py \
    --db /path/to/swarms/<name>/watchdog.db \
    --host 0.0.0.0 \
    --port 9100

Watchdog instances then send small JSON blobs via TCP to <host>:<port>.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import signal
import socket
import sqlite3
import sys
import time
from typing import Any

# Same table name as in watchdog
REPLICA_STATUS_TABLE = "replica_status"


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------


def open_db(path: Path) -> sqlite3.Connection:
    """
    Open (or create) the collector SQLite DB at `path`, ensuring the schema exists.
    Collector is single-writer, so locking issues should be minimal.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path.as_posix(), timeout=30.0)

    # Pragmas are best-effort; failures are non-fatal.
    try:
        conn.execute("PRAGMA synchronous=NORMAL")
    except sqlite3.OperationalError as e:
        print(
            f"collector: could not set synchronous=NORMAL ({e!r}); continuing.",
            file=sys.stderr,
        )
    try:
        conn.execute("PRAGMA busy_timeout=5000")
    except sqlite3.OperationalError as e:
        print(
            f"collector: could not set busy_timeout ({e!r}); continuing.",
            file=sys.stderr,
        )

    ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    """
    Create replica_status table if missing.
    """
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


def upsert_status(conn: sqlite3.Connection, payload: dict[str, Any]) -> None:
    """
    Insert or update a row in replica_status for a single (swarm_id, replica_id).

    Expected payload keys:
      - swarm_id (str)
      - replica_id (int)
      - node (str|None)
      - port (int|None)
      - pid (int|None)
      - state (str)
      - http_ready (bool|int)
      - exit_code (int|None)
      - exit_signal (int|None)
      - fail_reason (str|None)
      - agent_version (str|None)
    """
    swarm_id = str(payload.get("swarm_id", ""))
    try:
        replica_id = int(payload.get("replica_id", 0))
    except (TypeError, ValueError):
        print(
            f"collector: ignoring payload with invalid replica_id: {payload!r}",
            file=sys.stderr,
        )
        return

    if not swarm_id:
        # Ignore malformed packets quietly
        return

    node = payload.get("node")
    port = payload.get("port")
    pid = payload.get("pid")
    state = payload.get("state", "unknown")
    http_ready_raw = payload.get("http_ready", 0)
    http_ready = 1 if bool(http_ready_raw) else 0
    exit_code = payload.get("exit_code")
    exit_signal = payload.get("exit_signal")
    fail_reason = payload.get("fail_reason")
    agent_version = payload.get("agent_version") or "unknown"

    with conn:
        conn.execute(
            f"""
            INSERT INTO {REPLICA_STATUS_TABLE}
            (swarm_id, replica_id, node, port, pid, state, http_ready,
             exit_code, exit_signal, fail_reason, agent_version, last_seen)
            VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
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
                state,
                http_ready,
                exit_code,
                exit_signal,
                fail_reason,
                agent_version,
            ),
        )


# ---------------------------------------------------------------------------
# Collector loop (TCP)
# ---------------------------------------------------------------------------


def run_collector(db_path: Path, host: str, port: int) -> int:
    """
    Main collector loop.

    Listens on TCP socket at (host, port) for JSON messages from watchdogs.
    Each message describes the status of one replica, which is upserted into
    the replica_status table in the SQLite DB at db_path.

    Returns exit code (0=clean shutdown).
    """
    conn = open_db(db_path)

    # TCP server socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(128)  # now the kernel actually accepts connections
    sock.settimeout(1.0)  # allow periodic checks for shutdown

    hostname = socket.gethostname()
    print(
        f"collector: starting on host={hostname}, bind={host}:{port}, "
        f"db={db_path}, sock={sock.getsockname()}",
        file=sys.stderr,
    )

    stop_flag = {"stop": False}

    def _handle_sig(signum, frame):
        print(f"collector: received signal {signum}, shutting down...", file=sys.stderr)
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    try:
        while not stop_flag["stop"]:
            try:
                client_sock, addr = sock.accept()
            except TimeoutError:
                continue
            except OSError as e:
                print(f"collector: accept() error: {e!r}", file=sys.stderr)
                break

            with client_sock:
                try:
                    # Read entire payload from this connection
                    chunks: list[bytes] = []
                    client_sock.settimeout(2.0)
                    while True:
                        chunk = client_sock.recv(65535)
                        if not chunk:
                            break
                        chunks.append(chunk)

                    if not chunks:
                        print(
                            f"collector: empty payload from {addr}, ignoring",
                            file=sys.stderr,
                        )
                        continue

                    data = b"".join(chunks)
                    text = data.decode("utf-8", errors="replace").strip()
                    if not text:
                        print(
                            f"collector: whitespace-only payload from {addr}, ignoring",
                            file=sys.stderr,
                        )
                        continue

                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        print(
                            f"collector: received invalid JSON from {addr}, ignoring",
                            file=sys.stderr,
                        )
                        continue

                    if isinstance(payload, dict):
                        upsert_status(conn, payload)
                    else:
                        print(
                            f"collector: ignoring non-dict payload from {addr}",
                            file=sys.stderr,
                        )

                except (TimeoutError, OSError) as e:
                    print(
                        f"collector: socket error while reading from {addr}: {e!r}",
                        file=sys.stderr,
                    )
                except sqlite3.Error as e:
                    # Log but do not crash the collector.
                    print(
                        f"collector: SQLite error while upserting status: {e!r}",
                        file=sys.stderr,
                    )
                    time.sleep(0.1)

    finally:
        print("collector: cleaning up...", file=sys.stderr)
        with contextlib.suppress(Exception):
            conn.close()
            sock.close()

    print("collector: shutting down.", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Domyn-Swarm watchdog collector (single writer for replica_status)."
    )
    p.add_argument("--db", required=True, help="Path to swarm-local watchdog SQLite DB.")
    p.add_argument(
        "--host",
        default="0.0.0.0",
        help="IP/host to bind the TCP socket to (default: 0.0.0.0).",
    )
    p.add_argument(
        "--port",
        type=int,
        default=9100,
        help="TCP port to listen on (default: 9100).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = _parse_args(argv)
    return run_collector(Path(args.db), args.host, args.port)


if __name__ == "__main__":
    raise SystemExit(main())
