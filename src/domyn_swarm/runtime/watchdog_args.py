# Copyright 2025 iGenius S.p.A
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Iterable

from domyn_swarm.config.watchdog import WatchdogConfig as ConfigWatchdog


def build_watchdog_args(
    cfg: ConfigWatchdog,
    *,
    swarm_id: str,
    replica_id: int | str,
    node: str,
    port: int | str,
    log_dir: str,
    collector_address: str,
    agent_version: str | None = None,
    ray_enabled: bool | None = None,
    ray_expected_tp: int | str | None = None,
    ray_expected_workers: int | str | None = None,
) -> list[str]:
    """Build watchdog CLI args from a WatchdogConfig.

    Runtime-specific values (swarm_id, node, port, log_dir, collector address)
    must be passed by the caller. Optional ray fields can override config.
    """
    args = [
        "--swarm-id",
        swarm_id,
        "--replica-id",
        str(replica_id),
        "--node",
        node,
        "--port",
        str(port),
        "--probe-interval",
        str(cfg.probe_interval),
        "--http-path",
        cfg.http_path,
        "--http-timeout",
        str(cfg.http_timeout),
        "--restart-policy",
        cfg.restart_policy,
        "--readiness-timeout",
        str(cfg.readiness_timeout),
        "--unhealthy-restart-after",
        str(cfg.unhealthy_restart_after),
        "--max-restarts",
        str(cfg.max_restarts),
        "--restart-backoff",
        str(cfg.restart_backoff_initial),
        "--log-dir",
        log_dir,
        "--collector-address",
        collector_address,
    ]

    if agent_version:
        args.extend(["--agent-version", agent_version])

    ray_on = cfg.ray.enabled if ray_enabled is None else ray_enabled
    args.extend(["--ray-enabled", "1" if ray_on else "0"])

    expected_tp = cfg.ray.expected_tp if ray_expected_tp is None else ray_expected_tp
    if expected_tp is not None:
        args.extend(["--ray-expected-tp", str(expected_tp)])

    if ray_expected_workers is not None:
        args.extend(["--ray-expected-workers", str(ray_expected_workers)])

    args.extend(
        [
            "--ray-timeout",
            str(cfg.ray.probe_timeout_s),
            "--ray-grace",
            str(cfg.ray.status_grace_s),
            "--ray-probe-interval",
            str(cfg.ray.probe_interval_s),
        ]
    )

    return args


def args_to_str(args: Iterable[str]) -> str:
    return " ".join(args)
