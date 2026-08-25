# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import sys
import types

from domyn_swarm.runtime import watchdog as watchdog_mod


def test_ensure_leading_slash():
    """Adds a leading slash when missing."""
    assert watchdog_mod._ensure_leading_slash("v1/health") == "/v1/health"
    assert watchdog_mod._ensure_leading_slash("/v1/health") == "/v1/health"


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

    _http_failures, _http_ok_since, _ray_ok_since, _last_ray_probe, state, ready = (
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


# ---------------------------
# restart backoff
# ---------------------------


def test_restart_backoff_grows_exponentially():
    """Repeated restarts back off instead of hammering the scheduler at a fixed rate."""
    cfg = watchdog_mod.WatchdogConfig(restart_backoff_s=5.0, restart_backoff_max_s=60.0)

    delays = [watchdog_mod._restart_backoff_delay(cfg, attempt) for attempt in range(1, 5)]

    assert delays == [5.0, 10.0, 20.0, 40.0]


def test_restart_backoff_is_capped_by_the_configured_ceiling():
    """`restart_backoff_max` is the documented upper bound, so it must bind."""
    cfg = watchdog_mod.WatchdogConfig(restart_backoff_s=5.0, restart_backoff_max_s=30.0)

    delays = [watchdog_mod._restart_backoff_delay(cfg, attempt) for attempt in range(1, 6)]

    assert delays == [5.0, 10.0, 20.0, 30.0, 30.0]


# ---------------------------
# log level
# ---------------------------


def test_log_level_suppresses_lower_severity_diagnostics(caplog):
    """`watchdog.log_level: warning` must actually quieten the watchdog."""
    watchdog_mod._configure_logging("warning")

    with caplog.at_level(logging.DEBUG, logger=watchdog_mod.logger.name):
        watchdog_mod.logger.debug("a debug line")
        watchdog_mod.logger.info("an info line")
        watchdog_mod.logger.warning("a warning line")

    assert watchdog_mod.logger.level == logging.WARNING
    assert not watchdog_mod.logger.isEnabledFor(logging.INFO)
    assert watchdog_mod.logger.isEnabledFor(logging.WARNING)


def test_debug_level_lets_everything_through():
    """Raising verbosity is the documented way to debug a stuck replica."""
    watchdog_mod._configure_logging("debug")

    assert watchdog_mod.logger.isEnabledFor(logging.DEBUG)


def test_unknown_log_level_falls_back_to_info():
    """A bad value must not silence the watchdog entirely."""
    watchdog_mod._configure_logging("not-a-level")

    assert watchdog_mod.logger.level == logging.INFO


def test_logging_goes_to_stderr_only():
    """Watchdog output must land in the same captured stream as the child process."""
    watchdog_mod._configure_logging("info")

    handlers = watchdog_mod.logger.handlers
    assert handlers, "the watchdog configures its own handler; it has no root config"
    assert all(h.stream is sys.stderr for h in handlers)
    assert watchdog_mod.logger.propagate is False


def test_configure_logging_is_idempotent():
    """Re-configuring must not stack duplicate handlers."""
    watchdog_mod._configure_logging("info")
    watchdog_mod._configure_logging("debug")

    assert len(watchdog_mod.logger.handlers) == 1


def test_replica_exit_summary_bypasses_logging(capsys):
    """The summary line is parsed by callers, so it carries no prefix and is never filtered."""
    watchdog_mod._configure_logging("error")

    watchdog_mod._emit_exit_summary(0, {"exit_code": 0})

    err = capsys.readouterr().err
    assert err.strip() == 'watchdog[0]: {"exit_code":0}'
