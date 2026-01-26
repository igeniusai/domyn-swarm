from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import time


def get_free_port() -> int:
    """Return an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_port(host: str, port: int, timeout: float = 10.0) -> None:
    """Wait until (host, port) is accepting TCP connections or timeout."""

    def _can_connect(h: str, p: int) -> bool:
        try:
            with socket.create_connection((h, p), timeout=1.0):
                return True
        except OSError:
            return False

    deadline = time.time() + timeout
    while time.time() < deadline:
        if _can_connect(host, port):
            return
        time.sleep(0.1)

    raise TimeoutError(f"Port {host}:{port} did not open within {timeout} seconds")


def _read_replica_row_once(
    db_path: Path,
    swarm_id: str,
    replica_id: int,
) -> tuple[tuple | None, bool]:
    """
    Perform a single read attempt.

    Returns:
        (row, locked_flag)
        - row: the row tuple or None
        - locked_flag: True if the DB was locked/busy, False otherwise

    Raises:
        sqlite3.OperationalError for non-locking errors.
    """
    if not db_path.exists():
        return None, False

    try:
        conn = sqlite3.connect(db_path.as_posix(), timeout=1.0)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT state, http_ready, exit_code, fail_reason
                FROM replica_status
                WHERE swarm_id = ? AND replica_id = ?
                """,
                (swarm_id, replica_id),
            )
            row = cur.fetchone()
            return row, False
        finally:
            conn.close()
    except sqlite3.OperationalError as e:
        msg = str(e).lower()
        if "locked" in msg or "busy" in msg:
            # caller can decide to retry
            return None, True
        raise


def read_replica_row(
    db_path: Path,
    swarm_id: str,
    replica_id: int,
    *,
    attempts: int = 5,
    sleep_s: float = 0.05,
):
    """
    Safely read the single row for (swarm_id, replica_id) from replica_status,
    retrying on 'database is locked' errors.
    """
    for _ in range(attempts):
        row, locked = _read_replica_row_once(db_path, swarm_id, replica_id)
        if row is not None:
            return row
        if not locked:
            # No lock and no row → nothing to retry.
            return None
        time.sleep(sleep_s)

    return None


def wait_for_proc(proc: subprocess.Popen, timeout: float = 30.0) -> int:
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired as e:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        raise AssertionError("watchdog process did not exit in time") from e


def read_run_count(state_file: Path) -> int:
    if not state_file.exists():
        return 0
    text = state_file.read_text().strip()
    return int(text) if text else 0


def terminate_proc_with_logs(proc: subprocess.Popen, label: str = "watchdog") -> None:
    """
    Terminate a subprocess and safely collect its stdout/stderr without
    blocking on a pipe from a still-running process.

    - First try terminate + communicate(timeout=5).
    - On timeout, kill + communicate().
    - Print stderr (if any) for debugging.
    """
    if proc.poll() is not None:
        # Already exited: just drain pipes if they exist.
        try:
            _stdout, stderr = proc.communicate(timeout=1)
        except Exception:
            return
        if stderr:
            print(f"{label} stderr:\n{stderr}", file=sys.stderr)
        return

    try:
        proc.terminate()
        _stdout, stderr = proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        _stdout, stderr = proc.communicate()
    except Exception:
        # Last resort: don't let this crash the test.
        return

    if stderr:
        print(f"{label} stderr:\n{stderr}", file=sys.stderr)
