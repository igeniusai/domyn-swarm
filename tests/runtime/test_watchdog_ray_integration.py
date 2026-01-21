import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from .fixtures import *  # noqa: F403
from .helpers import get_free_port

RAY_FATAL_EXIT_CODE = 190


@pytest.fixture
def fake_ray_script() -> Path:
    """Path to the dedicated fake ray CLI."""
    return Path(__file__).with_name("fake_ray_cli.py")


@pytest.fixture
def fake_ray_bin(tmp_path: Path, fake_ray_script: Path) -> Path:
    """
    Create a disposable `bin` directory that exposes a `ray` executable,
    which simply forwards to `fake_ray_cli.py`.
    """
    bin_dir = tmp_path / "fake-ray-bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    wrapper = bin_dir / "ray"
    wrapper.write_text(
        f'#!/usr/bin/env bash\n"{sys.executable}" "{fake_ray_script.as_posix()}" "$@"\n'
    )
    wrapper.chmod(0o755)

    return bin_dir


def _parse_watchdog_summary(stdout: str, stderr: str, replica_id: int) -> dict:
    """
    Extract and parse the JSON summary line emitted by the watchdog, e.g.:

      watchdog[0]: {"exit_code":190,"should_restart":false,"fail_reason":"..."}

    We scan both stdout and stderr just in case.
    """
    combined = (stdout or "") + "\n" + (stderr or "")
    tag = f"watchdog[{replica_id}]"

    for line in reversed(combined.splitlines()):
        line = line.strip()
        if tag not in line:
            continue
        # Grab the JSON portion between the first '{' and the last '}'.
        if "{" not in line or "}" not in line:
            continue
        json_part = line[line.find("{") : line.rfind("}") + 1]
        try:
            return json.loads(json_part)
        except json.JSONDecodeError:
            continue

    raise AssertionError(
        f"Could not find valid watchdog summary JSON for {tag} in output.\n"
        f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
    )


@pytest.mark.integration
def test_ray_fatal_exit_is_not_restarted(
    collector_process,
    spawn_watchdog,
    fake_child_script: Path,
):
    """
    If the child exits with the Ray fatal exit code (190),
    the watchdog should:

      * propagate exit_code=190 in its JSON summary
      * set should_restart=False
      * not restart the child (child run counter == 1)
    """
    _db_path, collector_host, collector_port, _collector_proc = collector_process

    child_port = get_free_port()
    swarm_id = "test-swarm-ray-fatal"
    replica_id = 0
    node = "test-node-0"

    # Force the fake child to exit with 190 on the first run
    env_overrides = {
        "FAKE_CHILD_MODE": "ray_fatal_exit",
    }

    proc, state_file, _log_dir = spawn_watchdog(
        collector_host=collector_host,
        collector_port=collector_port,
        child_script=fake_child_script,
        swarm_id=swarm_id,
        replica_id=replica_id,
        node=node,
        child_port=child_port,
        extra_args=[
            # keep Ray behaviour enabled if your CLI needs it; if not,
            # you can omit these or adjust names to your real flags:
            # "--ray-enabled", "true",
            # "--ray-expected-tp", "4",
            # "--ray-expected-workers", "8",
            # we keep restart-policy=on-failure so generic non-zero
            # codes *would* restart, but 190 is special.
            "--restart-policy",
            "on-failure",
            "--max-restarts",
            "3",
        ],
        env_overrides=env_overrides,
    )

    # Wait for watchdog to exit; if it hangs, this will fail the test
    stdout, stderr = proc.communicate(timeout=30.0)

    if proc.returncode != 0:
        # This is still okay for the watchdog; we assert on the summary,
        # but dump logs for debugging if needed.
        print("=== WATCHDOG STDOUT ===", file=sys.stderr)
        print(stdout, file=sys.stderr)
        print("=== WATCHDOG STDERR ===", file=sys.stderr)
        print(stderr, file=sys.stderr)

    summary = _parse_watchdog_summary(stdout, stderr, replica_id=replica_id)

    # 1) Classification: we must see the Ray fatal exit code and no restart
    assert summary["exit_code"] == RAY_FATAL_EXIT_CODE
    assert summary["should_restart"] is False

    fail_reason = summary.get("fail_reason") or ""
    assert "exit_code=190" in fail_reason

    # 2) The child must have been started exactly once
    assert state_file.exists(), "fake child state file was not created"
    runs_raw = state_file.read_text().strip() or "0"
    runs = int(runs_raw)
    assert runs == 1, f"child was restarted unexpectedly (runs={runs})"


@pytest.mark.integration
def test_ray_capacity_insufficient_causes_no_restart_and_fatal_exit(
    tmp_path: Path,
    collector_process,
    spawn_watchdog,
    fake_child_script: Path,
    fake_ray_bin: Path,
):
    """
    If Ray is 'alive' but capacity is insufficient (too few GPUs/workers),
    the watchdog should:

      * exit with the dedicated 'Ray capacity' exit code (e.g. 191),
      * set should_restart=False,
      * never restart the child (run counter == 1).
    """
    _db_path, collector_host, collector_port, _collector_proc = collector_process

    child_port = get_free_port()
    swarm_id = "test-swarm-ray-capacity"
    replica_id = 0
    node = "test-node-0"

    # Here we model: Ray alive, but NOT enough capacity:
    #   FAKE_RAY_NODES          = 2
    #   FAKE_RAY_GPUS_PER_NODE  = 1
    # Expected TP = 4 and expected_workers = 4 -> capacity check must fail.
    env_overrides = {
        "FAKE_CHILD_MODE": "healthy",
        "PATH": f"{fake_ray_bin.as_posix()}:{os.environ.get('PATH', '')}",
        "FAKE_RAY_NODES": "2",
        "FAKE_RAY_GPUS_PER_NODE": "1.0",
    }

    proc, state_file, _log_dir = spawn_watchdog(
        collector_host=collector_host,
        collector_port=collector_port,
        child_script=fake_child_script,
        swarm_id=swarm_id,
        replica_id=replica_id,
        node=node,
        child_port=child_port,
        extra_args=[
            "--restart-policy",
            "on-failure",
            "--max-restarts",
            "3",
            "--probe-interval",
            "0.2",
            "--readiness-timeout",
            "10",
            "--unhealthy-restart-after",
            "5",
            # Ray-related CLI flags
            "--ray-enabled",
            "1",
            "--ray-expected-tp",
            "4",
            "--ray-expected-workers",
            "4",
            "--ray-probe-interval",
            "0.5",
            "--ray-grace",
            "1.0",
        ],
        env_overrides=env_overrides,
    )

    stdout, stderr = proc.communicate(timeout=20.0)

    # Debug if things go wrong
    print("=== WATCHDOG STDOUT ===", file=sys.stderr)
    print(stdout, file=sys.stderr)
    print("=== WATCHDOG STDERR ===", file=sys.stderr)
    print(stderr, file=sys.stderr)

    summary = _parse_watchdog_summary(stdout, stderr, replica_id=replica_id)

    # Capacity failure → special exit, no restart
    assert summary["exit_code"] == RAY_FATAL_EXIT_CODE
    assert summary["should_restart"] is False

    fail_reason = (summary.get("fail_reason") or "").lower()
    assert "ray" in fail_reason
    assert "capacity" in fail_reason or "placement group" in fail_reason

    # Child should only have been started once (no restart)
    assert state_file.exists(), "fake child state file was not created"
    runs_raw = state_file.read_text().strip() or "0"
    runs = int(runs_raw)
    assert runs == 1, f"child was restarted unexpectedly (runs={runs})"


@pytest.mark.timeout(60)
@pytest.mark.integration
def test_watchdog_exits_with_ray_special_code_when_capacity_drops_during_run(
    tmp_path: Path,
    fake_child_script: Path,
    fake_ray_bin: Path,
    collector_process,
    spawn_watchdog,
):
    """
    Scenario:
      - Ray cluster starts with sufficient capacity
        (enough ALIVE nodes & GPUs → ray_capacity_ok == True).
      - Fake child (vLLM) is healthy and /health passes.
      - Later, Ray capacity drops (fewer ALIVE nodes than expected),
        while ray status still reports ALIVE.
      - Watchdog should:
          * Mark the replica unhealthy due to Ray capacity.
          * Exit with the special Ray exit code (190).
          * Not treat this as a restartable error (i.e., intended for Slurm requeue).
    """

    _, coll_host, coll_port, _coll_proc = collector_process
    child_port = get_free_port()

    # Ray state file consumed by fake_ray_cli
    ray_state_file = tmp_path / "ray_state.json"

    # Initial state: enough capacity → 2 alive workers, 1 GPU each
    ray_state_file.write_text(
        json.dumps(
            {
                "alive_nodes": 2,
                "gpus_per_node": 1.0,
            }
        )
    )

    # Environment for the watchdog:
    # - FAKE_CHILD_MODE=healthy → fake child runs HTTP /health server
    # - PATH prefixed with directory containing fake_ray_cli ("ray" stub)
    # - FAKE_RAY_STATE_FILE → fake_ray_cli reads capacity from this file
    env_overrides = {
        "FAKE_CHILD_MODE": "healthy",
        "PATH": f"{fake_ray_bin.as_posix()}:{os.environ.get('PATH', '')}",
        "FAKE_RAY_STATE_FILE": ray_state_file.as_posix(),
    }

    extra_args = [
        "--ray-enabled",
        "1",
        "--ray-expected-tp",
        "1",
        "--ray-expected-workers",
        "2",
        "--ray-probe-interval",
        "0.5",
        "--ray-grace",
        "0.0",
        "--ray-timeout",
        "1.0",
    ]

    proc, _state_file, _log_dir = spawn_watchdog(
        collector_host=coll_host,
        collector_port=coll_port,
        child_script=fake_child_script,
        swarm_id="swarm-ray-capacity-drop",
        replica_id=0,
        node="nodeA",
        child_port=child_port,
        extra_args=extra_args,
        env_overrides=env_overrides,
    )

    try:
        # 1) Let the watchdog + fake child reach RUNNING / healthy.
        #    With small probe intervals + readiness_timeout in the fixture,
        #    a short sleep is enough; we don't strictly need to hit the DB here.
        time.sleep(3.0)

        # 2) Now simulate Ray losing capacity:
        #    Drop alive_nodes from 2 → 1, while fake_ray_cli still
        #    reports "status" as success.
        ray_state_file.write_text(
            json.dumps(
                {
                    "alive_nodes": 1,  # < expected_workers (=2)
                    "gpus_per_node": 1.0,
                }
            )
        )

        # 3) Wait for watchdog to notice the capacity drop on the next
        #    Ray probe cycle and exit with the special code.
        exit_code = proc.wait(timeout=30.0)

        # (Optional debugging aid: if needed, print stderr on failure)
        if exit_code != RAY_FATAL_EXIT_CODE and proc.stderr:
            err = proc.stderr.read()
            print("watchdog stderr:\n", err)

        # 4) Assert: watchdog exited with the Ray-special exit code.
        assert exit_code == RAY_FATAL_EXIT_CODE, (
            f"Expected watchdog to exit with Ray special code "
            f"{RAY_FATAL_EXIT_CODE}, got {exit_code}"
        )

    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
