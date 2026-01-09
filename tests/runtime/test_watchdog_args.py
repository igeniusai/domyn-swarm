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
