# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from domyn_swarm.config.watchdog import WatchdogConfig
from domyn_swarm.runtime.watchdog_args import args_to_str, build_watchdog_args


def test_build_watchdog_args_from_config():
    cfg = WatchdogConfig()
    args = build_watchdog_args(
        cfg,
        swarm_id="swarm",
        replica_id=1,
        node="node1",
        port=8000,
        log_dir="/tmp/logs",
        collector_address="127.0.0.1:9100",
        agent_version="v1",
        ray_expected_tp=8,
        ray_expected_workers=2,
    )

    assert "--probe-interval" in args
    assert str(cfg.probe_interval) in args
    assert "--http-timeout" in args
    assert str(cfg.http_timeout) in args
    assert "--restart-policy" in args
    assert cfg.restart_policy in args
    assert "--restart-backoff" in args
    assert str(cfg.restart_backoff_initial) in args
    assert "--ray-expected-tp" in args
    assert "8" in args
    assert "--ray-expected-workers" in args
    assert "2" in args


def test_build_watchdog_args_rendered_string():
    cfg = WatchdogConfig()
    rendered = args_to_str(
        build_watchdog_args(
            cfg,
            swarm_id="swarm",
            replica_id="1",
            node="node1",
            port="8000",
            log_dir="/tmp/logs",
            collector_address="127.0.0.1:9100",
            agent_version="v1",
            ray_enabled=False,
        )
    )

    assert "--swarm-id swarm" in rendered
    assert "--replica-id 1" in rendered
    assert "--node node1" in rendered
    assert "--port 8000" in rendered
    assert "--log-dir /tmp/logs" in rendered
    assert "--collector-address 127.0.0.1:9100" in rendered
    assert "--agent-version v1" in rendered


def _args_value(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def test_log_level_is_forwarded_to_the_watchdog():
    """`watchdog.log_level` is documented as the way to raise watchdog verbosity."""
    cfg = WatchdogConfig(log_level="debug")
    args = build_watchdog_args(
        cfg,
        swarm_id="swarm",
        replica_id=1,
        node="node1",
        port=8000,
        log_dir="/tmp/logs",
        collector_address="127.0.0.1:9100",
    )

    assert _args_value(args, "--log-level") == "debug"


def test_restart_backoff_max_is_forwarded_to_the_watchdog():
    """`restart_backoff_max` bounds the exponential backoff, so it must reach the process."""
    cfg = WatchdogConfig(restart_backoff_initial=3, restart_backoff_max=45)
    args = build_watchdog_args(
        cfg,
        swarm_id="swarm",
        replica_id=1,
        node="node1",
        port=8000,
        log_dir="/tmp/logs",
        collector_address="127.0.0.1:9100",
    )

    assert _args_value(args, "--restart-backoff") == "3"
    assert _args_value(args, "--restart-backoff-max") == "45"


def test_every_config_field_is_either_forwarded_or_explicitly_local():
    """Guard against silently adding a WatchdogConfig field that nothing consumes."""
    # `enabled` gates whether the watchdog is launched at all, in the sbatch template.
    handled_outside_argv = {"enabled"}
    rendered = args_to_str(
        build_watchdog_args(
            WatchdogConfig(),
            swarm_id="swarm",
            replica_id=1,
            node="node1",
            port=8000,
            log_dir="/tmp/logs",
            collector_address="127.0.0.1:9100",
        )
    )

    unconsumed = []
    for field in WatchdogConfig.model_fields:
        if field in handled_outside_argv or field == "ray":
            continue
        flag = "--" + field.replace("_", "-")
        # Accept either the exact flag or a shortened spelling of it.
        if flag not in rendered and not any(
            part.startswith("--") and field.replace("_", "-").startswith(part[2:])
            for part in rendered.split()
        ):
            unconsumed.append(field)

    assert not unconsumed, f"WatchdogConfig fields never reach the watchdog: {unconsumed}"
