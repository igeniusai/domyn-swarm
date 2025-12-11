import os
from pathlib import Path
import subprocess
import sys

import pytest

from .helpers import (
    get_free_port,
    wait_for_port,
)


@pytest.fixture
def fake_child_script() -> Path:
    """
    Return the path to the dedicated fake watchdog child script.

    We assume it's located next to this test file as `fake_watchdog_child.py`:
      tests/runtime/fake_watchdog_child.py
    """
    return Path(__file__).with_name("fake_watchdog_child.py")


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Location of the collector's SQLite DB."""
    return tmp_path / "watchdog.db"


@pytest.fixture
def collector_process(
    db_path: Path,
) -> tuple[Path, str, int, subprocess.Popen]:
    """
    Start the collector as a subprocess:

      python -m domyn_swarm.runtime.collector --db ... --host 0.0.0.0 --port ...

    Wait until it's actually listening and yield (db_path, host, port, proc).
    """
    host = "127.0.0.1"
    port = get_free_port()

    cmd = [
        sys.executable,
        "-m",
        "domyn_swarm.runtime.collector",
        "--db",
        db_path.as_posix(),
        "--host",
        host,
        "--port",
        str(port),
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait for socket to be open
    try:
        wait_for_port(host, port, timeout=10.0)
    except Exception:
        # If it fails, dump some stderr for debugging
        if proc.poll() is not None and proc.stderr is not None:
            stderr = proc.stderr.read()
            print("collector stderr:\n", stderr, file=sys.stderr)
        raise

    yield db_path, host, port, proc

    # Teardown
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


@pytest.fixture
def spawn_watchdog(tmp_path: Path):
    """
    Return a helper that spawns the watchdog process with given parameters
    and returns (process, state_file, log_dir).

    This uses:
      - --collector-address host:port
      - a dedicated state file per replica to track restart attempts
    """

    def _spawn_watchdog(
        *,
        collector_host: str,
        collector_port: int,
        child_script: Path,
        swarm_id: str,
        replica_id: int,
        node: str,
        child_port: int,
        extra_args: list[str] | None = None,
        env_overrides: dict[str, str] | None = None,
    ) -> tuple[subprocess.Popen, Path, Path]:
        log_dir = tmp_path / f"logs-replica-{replica_id}"
        log_dir.mkdir(parents=True, exist_ok=True)
        state_file = tmp_path / f"child_state_{replica_id}.txt"

        cmd: list[str] = [
            sys.executable,
            "-m",
            "domyn_swarm.runtime.watchdog",
            "--swarm-id",
            swarm_id,
            "--replica-id",
            str(replica_id),
            "--node",
            node,
            "--port",
            str(child_port),
            "--log-dir",
            log_dir.as_posix(),
            "--collector-address",
            f"{collector_host}:{collector_port}",
            # Fast, test-friendly defaults (can be overridden by extra_args):
            "--probe-interval",
            "0.2",
            "--http-timeout",
            "0.5",
            "--unhealthy-http-failures",
            "1",
            "--restart-policy",
            "on-failure",
            "--restart-backoff",
            "0.5",
            "--max-restarts",
            "1",
            "--unhealthy-restart-after",
            "2",
            "--readiness-timeout",
            "10",
        ]

        if extra_args:
            cmd.extend(extra_args)

        # Child command (fake vLLM) after `--`
        cmd.extend(
            [
                "--",
                sys.executable,
                child_script.as_posix(),
                "--port",
                str(child_port),
                "--state-file",
                state_file.as_posix(),
            ]
        )

        env = os.environ.copy()
        if env_overrides:
            env.update(env_overrides)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        return proc, state_file, log_dir

    return _spawn_watchdog
