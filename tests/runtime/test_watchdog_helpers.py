import json
import types

from domyn_swarm.runtime import watchdog as watchdog_mod


def test_ensure_leading_slash():
    """Adds a leading slash when missing."""
    assert watchdog_mod._ensure_leading_slash("v1/models") == "/v1/models"
    assert watchdog_mod._ensure_leading_slash("/v1/models") == "/v1/models"


def test_classify_fail_reason_from_log():
    """Detects non-retryable placement group failures."""
    reason, retryable = watchdog_mod.classify_fail_reason_from_log(
        "Cannot provide a placement group"
    )
    assert reason == "ray_pg_insufficient_capacity"
    assert retryable is False


def test_build_fail_reason_includes_log_tail(tmp_path):
    """Includes exit info and log tail in failure reason."""
    log_path = tmp_path / "vllm.log"
    log_path.write_text("line1\nCannot provide a placement group\nline3\n")
    reason, retryable = watchdog_mod.build_fail_reason(
        exit_code=137,
        exit_signal=None,
        log_path=str(log_path),
        restart_attempt=1,
        max_restarts=3,
    )
    assert "exit_code=137" in reason
    assert "ray_pg_insufficient_capacity" in reason
    assert retryable is False


def test_should_restart_policies():
    """Honors restart policy and fatal exit code."""
    cfg = watchdog_mod.WatchdogConfig(restart_policy="never")
    assert watchdog_mod._should_restart(1, cfg, restart_count=0) is False
    cfg.restart_policy = "on-failure"
    assert watchdog_mod._should_restart(0, cfg, restart_count=0) is False
    assert watchdog_mod._should_restart(1, cfg, restart_count=0) is True
    assert watchdog_mod._should_restart(watchdog_mod.RAY_FATAL_EXIT_CODE, cfg, 0) is False


def test_ray_capacity_ok(monkeypatch):
    """Checks GPU and worker capacity from ray status output."""
    nodes = [
        {"state": "ALIVE", "resources_total": {"GPU": 2}},
        {"state": "ALIVE", "resources_total": {"GPU": 2}},
    ]
    cp = types.SimpleNamespace(returncode=0, stdout=json.dumps(nodes))
    monkeypatch.setattr(watchdog_mod, "_run_cmd", lambda *args, **kwargs: cp)
    assert watchdog_mod._ray_capacity_ok([], expected_tp=2, expected_workers=2) is True
    assert watchdog_mod._ray_capacity_ok([], expected_tp=8, expected_workers=2) is False


def test_probe_and_update_marks_running(monkeypatch):
    """Marks running when HTTP is ready and ray is disabled."""
    cfg = watchdog_mod.WatchdogConfig()
    cfg.ray.enabled = False
    meta = watchdog_mod.ReplicaMeta(swarm_id="s", replica_id=0, node="n", port=1)
    monkeypatch.setattr(watchdog_mod, "_check_http", lambda *args, **kwargs: True)
    monkeypatch.setattr(watchdog_mod, "send_status", lambda *args, **kwargs: None)

    http_failures, http_ok_since, ray_ok_since, last_ray_probe, state, ready = (
        watchdog_mod._probe_and_update(
            "localhost:1",
            meta,
            cfg,
            pid=1,
            http_failures=0,
            http_ok_since=None,
            ray_ok_since=None,
            last_ray_probe=0.0,
            ray_prefix=[],
        )
    )
    assert state == watchdog_mod.ReplicaState.RUNNING
    assert ready is True
