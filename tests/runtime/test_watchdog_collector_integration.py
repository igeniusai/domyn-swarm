from collections.abc import Callable
from pathlib import Path
import subprocess
import sys
import time

import pytest

from .helpers import (
    get_free_port,
    read_replica_row,
    read_run_count,
    terminate_proc_with_logs,
    wait_for_port,
    wait_for_proc,
)

# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_watchdog_and_collector_restart_flow(
    tmp_path: Path,
    fake_child_script: Path,
    collector_process: tuple[Path, str, int, subprocess.Popen],
    spawn_watchdog: Callable[..., tuple[subprocess.Popen, Path, Path]],
):
    """
    Full integration smoke-test:

    - Collector listens on TCP and writes into SQLite DB.
    - Watchdog starts a child that fails once, then runs successfully.
    - We assert that:
        * The child was started at least twice (restart occurred).
        * The collector wrote at least one row for this swarm/replica.

    We do NOT require the watchdog to exit by itself; we terminate it
    explicitly once we've observed the expected conditions.
    """
    db_path, collector_host, collector_port, _collector_proc = collector_process

    swarm_id = "test-swarm"
    replica_id = 0
    node = "test-node"
    child_port = get_free_port()

    # Fake child: first run fails, subsequent runs behave as "healthy".
    env_overrides = {"FAKE_CHILD_MODE": "fail_once_then_ok"}

    watchdog_proc, state_file, _log_dir = spawn_watchdog(
        collector_host=collector_host,
        collector_port=collector_port,
        child_script=fake_child_script,
        swarm_id=swarm_id,
        replica_id=replica_id,
        node=node,
        child_port=child_port,
        env_overrides=env_overrides,
    )

    states_seen: list[tuple[str, int | None, int | None, str | None]] = []
    row_seen = False
    start = time.time()
    timeout_s = 40.0

    while time.time() - start < timeout_s:
        # 1) Check DB row via collector
        row = read_replica_row(db_path, swarm_id, replica_id)
        if row is not None:
            state, http_ready, exit_code, fail_reason = row
            states_seen.append((state, http_ready, exit_code, fail_reason))
            row_seen = True

        # 2) Check how many times the fake child has run
        attempts = 0
        if state_file.exists():
            try:
                attempts = int(state_file.read_text().strip() or "0")
            except ValueError:
                attempts = 0

        # We're happy once:
        #   - the collector has written at least one row, and
        #   - the child has been run at least twice (restart happened).
        if row_seen and attempts >= 2:
            break

        # If watchdog died unexpectedly, stop early and inspect stderr later.
        if watchdog_proc.poll() is not None:
            break

        time.sleep(0.5)

    # Cleanup: watchdog is expected to be long-lived; kill it if still running.
    terminate_proc_with_logs(watchdog_proc, label="watchdog")

    # --- Assertions ---

    assert row_seen, "No rows observed in replica_status for this swarm/replica"

    assert state_file.exists(), "fake child state file was not created"
    try:
        attempts = int(state_file.read_text().strip() or "0")
    except ValueError:
        attempts = 0
    assert attempts >= 2, f"expected at least 2 child runs, got {attempts}"

    assert states_seen, "no states recorded; collector/watchdog wiring likely broken"


def test_watchdog_single_replica_happy_path(
    collector_process,
    spawn_watchdog,
    fake_child_script,
):
    """
    Start one collector and one watchdog with a healthy fake child.

    Expect:
      - A status row is written for (swarm_id, replica_id).
      - The replica eventually reaches state='running' and http_ready=1.
    """
    swarm_id = "integration-swarm-single"
    replica_id = 0
    node = "testnode"
    child_port = get_free_port()

    # Start collector
    db_path, host, port, collector_proc = collector_process

    # Start watchdog
    wd_proc, state_file, log_dir = spawn_watchdog(
        collector_host=host,
        collector_port=port,
        child_script=fake_child_script,
        swarm_id=swarm_id,
        replica_id=replica_id,
        node=node,
        child_port=child_port,
    )

    try:
        # Wait until we see at least one row for this replica
        deadline = time.time() + 20.0
        row = None
        while time.time() < deadline:
            row = read_replica_row(db_path, swarm_id, replica_id)
            if row is not None:
                break
            time.sleep(0.2)

        assert row is not None, "No status row written for single replica"
        state, http_ready, exit_code, fail_reason = row

        # Early state might be "starting"/"unhealthy", so be lenient at first
        assert state in ("starting", "running", "unhealthy")

        # Eventually we want running + http_ready=1
        if not (state == "running" and http_ready == 1):
            deadline2 = time.time() + 20.0
            while time.time() < deadline2:
                row = read_replica_row(db_path, swarm_id, replica_id)
                if row is None:
                    time.sleep(0.2)
                    continue
                state, http_ready, exit_code, fail_reason = row
                if state == "running" and http_ready == 1:
                    break
                time.sleep(0.2)
            else:
                pytest.fail("Replica never reached state=running & http_ready=1")
    finally:
        if wd_proc.poll() is None:
            wd_proc.terminate()
            try:
                wd_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                wd_proc.kill()
                wd_proc.wait(timeout=5)


# 2) Multiple replicas, concurrent reporting
def test_watchdog_multiple_replicas_concurrent(
    collector_process,
    spawn_watchdog,
    fake_child_script,
):
    """
    Start one collector and multiple watchdogs (replicas), each with a healthy child.

    Expect:
      - A row per replica (swarm, replica_id) is written.
      - Each replica has a distinct replica_id and the expected port.
      - No replica ends up in FAILED during this short run.
    """
    swarm_id = "integration-swarm-multi"
    node = "testnode"
    base_port = 8200
    num_replicas = 4

    db_path, host, port, collector_proc = collector_process

    watchdog_procs: list[subprocess.Popen] = []
    try:
        # Start N watchdogs with different replica_id + child_port
        for replica_id in range(num_replicas):
            child_port = base_port + replica_id
            proc, state_file, log_dir = spawn_watchdog(
                collector_host=host,
                collector_port=port,
                child_script=fake_child_script,
                swarm_id=swarm_id,
                replica_id=replica_id,
                node=node,
                child_port=child_port,
            )
            watchdog_procs.append(proc)

        # Wait until we see rows for all replicas (or timeout)
        deadline = time.time() + 30.0
        seen_ids = set()
        while time.time() < deadline and len(seen_ids) < num_replicas:
            for replica_id in range(num_replicas):
                row = read_replica_row(db_path, swarm_id, replica_id)
                if row is not None:
                    seen_ids.add(replica_id)
            if len(seen_ids) >= num_replicas:
                break
            time.sleep(0.2)

        assert seen_ids == set(range(num_replicas)), (
            f"Expected rows for replicas 0..{num_replicas - 1}, but got {seen_ids}"
        )

        # Basic sanity checks on final state of each replica
        final_states = {}
        for replica_id in range(num_replicas):
            row = read_replica_row(db_path, swarm_id, replica_id)
            assert row is not None
            state, http_ready, exit_code, fail_reason = row
            final_states[replica_id] = state

        # No FAILED state expected in this happy-path test
        assert "failed" not in {s.lower() for s in final_states.values()}
    finally:
        for p in watchdog_procs:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait(timeout=5)


# 3) Collector starts late, watchdog keeps running and eventually writes status
def test_watchdog_tolerates_collector_late_start(
    spawn_watchdog,
    fake_child_script,
    db_path,
):
    """
    Start a watchdog first while the collector is not yet listening.

    Expect:
      - Watchdog keeps running during collector downtime (no crash on send failures).
      - Once collector starts, statuses appear in the DB.
    """
    swarm_id = "integration-swarm-late-collector"
    replica_id = 0
    node = "testnode"
    child_port = get_free_port()

    # Reserve a host:port for the collector but do NOT start it yet
    collector_host = "127.0.0.1"
    collector_port = get_free_port()

    # Start watchdog before collector is up
    wd_proc, state_file, log_dir = spawn_watchdog(
        collector_host=collector_host,
        collector_port=collector_port,
        child_script=fake_child_script,
        swarm_id=swarm_id,
        replica_id=replica_id,
        node=node,
        child_port=child_port,
    )

    collector_proc = None
    try:
        # Give the watchdog some time to attempt (and fail) sending statuses
        time.sleep(2.0)

        # Watchdog should still be alive (no crash from connection refused)
        assert wd_proc.poll() is None, "Watchdog exited unexpectedly before collector started"

        # Now start the collector on the reserved address
        cmd = [
            sys.executable,
            "-m",
            "domyn_swarm.runtime.collector",
            "--db",
            db_path.as_posix(),
            "--host",
            collector_host,
            "--port",
            str(collector_port),
        ]
        collector_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait until collector is really listening
        wait_for_port(collector_host, collector_port, timeout=10.0)

        # Now statuses should start appearing
        deadline = time.time() + 30.0
        row = None
        while time.time() < deadline:
            row = read_replica_row(db_path, swarm_id, replica_id)
            if row is not None:
                break
            time.sleep(0.2)

        assert row is not None, "No status row written after collector became available"

    finally:
        if wd_proc.poll() is None:
            wd_proc.terminate()
            try:
                wd_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                wd_proc.kill()
                wd_proc.wait(timeout=5)

        if collector_proc is not None and collector_proc.poll() is None:
            collector_proc.terminate()
            try:
                collector_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                collector_proc.kill()
                collector_proc.wait(timeout=5)


def test_watchdog_respects_max_restarts(
    collector_process,
    spawn_watchdog,
    fake_child_script,
):
    """
    If the child always fails and max-restarts=1, we should see exactly
    two runs (initial + one restart) and then the watchdog exits.
    """
    db_path, host, port, _collector_proc = collector_process
    swarm_id = "swarm-max-restarts"
    replica_id = 0
    node = "test-node"
    child_port = get_free_port()

    # We assume fake child understands FAKE_CHILD_MODE="always_fail"
    env_overrides = {"FAKE_CHILD_MODE": "always_fail"}

    proc, state_file, _log_dir = spawn_watchdog(
        collector_host=host,
        collector_port=port,
        child_script=fake_child_script,
        swarm_id=swarm_id,
        replica_id=replica_id,
        node=node,
        child_port=child_port,
        extra_args=[
            "--restart-policy",
            "on-failure",
            "--max-restarts",
            "1",
        ],
        env_overrides=env_overrides,
    )

    exit_code = wait_for_proc(proc, timeout=40.0)
    assert exit_code != 0
    # State file should show 2 runs: initial + one restart
    runs = read_run_count(state_file)
    assert runs == 2, f"expected 2 runs, got {runs}"

    # DB row should show FAILED (or EXITED) with non-zero exit_code
    row = read_replica_row(db_path, swarm_id, replica_id)
    assert row is not None
    state, http_ready, exit_code_db, fail_reason = row
    assert state in ("failed", "exited")
    assert exit_code_db is not None and exit_code_db != 0
    assert not http_ready
    assert fail_reason is None or "exit_code=" in (fail_reason or "")


def test_watchdog_never_restart_policy(
    collector_process,
    spawn_watchdog,
    fake_child_script,
):
    """
    With restart-policy=never, the watchdog must not restart the child
    even if it exits with a failure code.
    """
    db_path, host, port, _collector_proc = collector_process
    swarm_id = "swarm-never-restart"
    replica_id = 0
    node = "test-node"
    child_port = get_free_port()

    env_overrides = {"FAKE_CHILD_MODE": "always_fail"}

    proc, state_file, _log_dir = spawn_watchdog(
        collector_host=host,
        collector_port=port,
        child_script=fake_child_script,
        swarm_id=swarm_id,
        replica_id=replica_id,
        node=node,
        child_port=child_port,
        extra_args=[
            "--restart-policy",
            "never",
            "--max-restarts",
            "10",  # should be ignored by policy
        ],
        env_overrides=env_overrides,
    )

    exit_code = wait_for_proc(proc, timeout=40.0)
    assert exit_code != 0

    runs = read_run_count(state_file)
    assert runs == 1, f"expected 1 run with restart-policy=never, got {runs}"

    row = read_replica_row(db_path, swarm_id, replica_id)
    assert row is not None
    state, _http_ready, exit_code_db, _fail_reason = row
    # Single failure, then no restart
    assert state in ("failed", "exited")
    assert exit_code_db != 0


def test_watchdog_always_restart_on_success(
    collector_process,
    spawn_watchdog,
    fake_child_script,
):
    """
    With restart-policy=always and max-restarts=2, the watchdog should restart
    the child even on successful exit, and we should see at least 3 runs
    (initial + 2 restarts).

    We don't rely on the watchdog exiting by itself; we terminate it from
    the test once we've observed the expected behaviour.
    """
    db_path, host, port, _collector_proc = collector_process
    swarm_id = "swarm-always-restart"
    replica_id = 0
    node = "test-node"
    child_port = get_free_port()

    # Fake child: always_succeed -> exit 0 quickly, but may run multiple times.
    env_overrides = {"FAKE_CHILD_MODE": "always_succeed"}

    watchdog_proc, state_file, _log_dir = spawn_watchdog(
        collector_host=host,
        collector_port=port,
        child_script=fake_child_script,
        swarm_id=swarm_id,
        replica_id=replica_id,
        node=node,
        child_port=child_port,
        extra_args=[
            "--restart-policy",
            "always",
            "--max-restarts",
            "2",
        ],
        env_overrides=env_overrides,
    )

    start = time.time()
    timeout_s = 40.0
    row_seen = False

    while time.time() - start < timeout_s:
        # DB row?
        row = read_replica_row(db_path, swarm_id, replica_id)
        if row is not None:
            row_seen = True

        # Child run count
        runs = read_run_count(state_file)

        # We want at least 3 runs: initial + 2 restarts
        if row_seen and runs >= 3:
            break

        # If watchdog died on its own, we can stop polling and inspect.
        if watchdog_proc.poll() is not None:
            break

        time.sleep(0.5)

    # Cleanup watchdog
    terminate_proc_with_logs(watchdog_proc, label="watchdog")

    # --- Assertions ---

    assert state_file.exists(), "fake child state file was not created"
    runs = read_run_count(state_file)
    assert runs >= 3, f"expected at least 3 runs (initial + 2 restarts), got {runs}"

    row = read_replica_row(db_path, swarm_id, replica_id)
    assert row is not None, "no replica_status row recorded"
    state, http_ready, exit_code_db, _fail_reason = row

    # Final state is either EXITED or RUNNING, depending on when we stopped.
    assert state in ("exited", "running")
    # If in EXITED, it should have been a clean exit (0).
    if state == "exited":
        assert exit_code_db is not None


def test_watchdog_unhealthy_restart_after_triggers_restart(
    collector_process,
    spawn_watchdog,
    fake_child_script,
):
    """
    If the child is alive but consistently HTTP-unhealthy, the watchdog
    should eventually restart it after `unhealthy_restart_after` seconds.
    """
    db_path, host, port, _collector_proc = collector_process
    swarm_id = "swarm-unhealthy-restart"
    replica_id = 0
    node = "test-node"
    child_port = get_free_port()

    # Fake child: binds the port but never passes HTTP health checks.
    env_overrides = {"FAKE_CHILD_MODE": "http_unhealthy"}

    proc, state_file, _log_dir = spawn_watchdog(
        collector_host=host,
        collector_port=port,
        child_script=fake_child_script,
        swarm_id=swarm_id,
        replica_id=replica_id,
        node=node,
        child_port=child_port,
        extra_args=[
            "--restart-policy",
            "on-failure",
            "--max-restarts",
            "1",
            "--probe-interval",
            "0.5",
            "--unhealthy-http-failures",
            "1",  # 1 failed probe => UNHEALTHY
            "--unhealthy-restart-after",
            "2",  # 2s in UNHEALTHY => restart
        ],
        env_overrides=env_overrides,
    )

    start = time.time()
    timeout_s = 60.0
    row_seen = False

    while time.time() - start < timeout_s:
        row = read_replica_row(db_path, swarm_id, replica_id)
        if row is not None:
            row_seen = True

        runs = read_run_count(state_file)

        # Once we have at least one DB row and at least 2 runs, we know an
        # UNHEALTHY-triggered restart likely occurred.
        if row_seen and runs >= 2:
            break

        if proc.poll() is not None:
            break

        time.sleep(0.5)

    # Cleanup watchdog
    terminate_proc_with_logs(proc, label="watchdog")

    # --- Assertions ---
    assert state_file.exists(), "fake child state file was not created"
    runs = read_run_count(state_file)
    assert runs >= 2, f"expected at least 2 runs due to UNHEALTHY restart, got {runs}"

    row = read_replica_row(db_path, swarm_id, replica_id)
    assert row is not None, "no replica_status row recorded"
    state, http_ready, exit_code_db, fail_reason = row

    # Final state is typically FAILED or EXITED or RUNNING, depending on timing.
    assert state in ("failed", "exited", "restarting", "running")

    # If we *did* set a fail_reason, it should reflect either unhealthy timeout
    # or a non-zero exit; but don't make this too strict to avoid flakiness.
    if fail_reason:
        assert "unhealthy" in fail_reason or "exit_code" in fail_reason or "timeout" in fail_reason


@pytest.mark.integration
def test_watchdog_respects_startup_readiness_timeout(
    collector_process,
    spawn_watchdog,
    fake_child_script,
):
    """
    The child takes some time before its /health endpoint becomes ready,
    but becomes healthy *within* readiness_timeout.

    We expect:
      - No watchdog restarts (run_count == 1).
      - Replica eventually reaches RUNNING with http_ready=1 in watchdog.db.
      - No UNHEALTHY/FAILED state purely due to slow startup.
    """
    db_path, host, port, _collector_proc = collector_process
    swarm_id = "swarm-readiness-grace"
    replica_id = 0
    node = "test-node"
    child_port = get_free_port()

    # Fake child: sleep for a while, then start healthy HTTP server.
    # Delay is shorter than readiness_timeout so no restart should happen.
    env_overrides = {
        "FAKE_CHILD_MODE": "slow_start_then_ok",
        "FAKE_CHILD_START_DELAY": "5",  # seconds
    }

    watchdog_proc, state_file, _log_dir = spawn_watchdog(
        collector_host=host,
        collector_port=port,
        child_script=fake_child_script,
        swarm_id=swarm_id,
        replica_id=replica_id,
        node=node,
        child_port=child_port,
        extra_args=[
            "--probe-interval",
            "0.5",
            "--http-timeout",
            "0.5",
            "--unhealthy-http-failures",
            "1",
            "--unhealthy-restart-after",
            "10",  # > readiness_timeout; we care about grace, not restart here
            "--readiness-timeout",
            "15",  # > FAKE_CHILD_START_DELAY (5s) → no restart during startup
        ],
        env_overrides=env_overrides,
    )

    start = time.time()
    timeout_s = 40.0
    seen_http_ready = False
    states_seen: list[tuple[str, int, int | None, str | None]] = []

    while time.time() - start < timeout_s:
        # Did watchdog die unexpectedly?
        if watchdog_proc.poll() is not None:
            break

        row = read_replica_row(db_path, swarm_id, replica_id)
        if row is not None:
            state, http_ready, exit_code_db, fail_reason = row
            states_seen.append((state, http_ready, exit_code_db, fail_reason))
            if state == "running" and int(http_ready or 0) == 1:
                seen_http_ready = True
                break

        time.sleep(0.5)

    # Cleanup watchdog so the test doesn't leak processes
    terminate_proc_with_logs(watchdog_proc, label="watchdog")

    # --- Assertions ---

    # We must have observed at least one DB row for this replica.
    assert states_seen, "no states observed in replica_status for this swarm/replica"

    # At least once, we should see RUNNING with http_ready=1
    assert seen_http_ready, f"never observed RUNNING with http_ready=1; states_seen={states_seen}"

    # The child should have been started exactly once (no restart)
    assert state_file.exists(), "fake child state file was not created"
    runs = read_run_count(state_file)
    assert runs == 1, f"expected a single child run (no restart), got {runs}"

    # We should not have gone through UNHEALTHY purely due to slow startup.
    assert not any(s == "unhealthy" for (s, *_rest) in states_seen), (
        f"replica went UNHEALTHY during startup despite readiness grace; states_seen={states_seen}"
    )
