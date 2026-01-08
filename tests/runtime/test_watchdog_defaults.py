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

from domyn_swarm.runtime.watchdog import _parse_args


def test_watchdog_default_args(tmp_path):
    args = _parse_args(
        [
            "--swarm-id",
            "swarm",
            "--replica-id",
            "0",
            "--node",
            "node0",
            "--port",
            "8000",
            "--log-dir",
            str(tmp_path),
        ]
    )

    assert args.http_path == "/health"
    assert args.probe_interval == 30.0
    assert args.http_timeout == 2.0
    assert args.readiness_timeout == 600.0
    assert args.restart_policy == "on-failure"
    assert args.restart_backoff == 5.0
    assert args.max_restarts == 3
    assert args.unhealthy_restart_after == 120.0
    assert args.ray_probe_interval == 30.0
