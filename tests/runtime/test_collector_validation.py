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

from domyn_swarm.runtime.collector import _normalize_payload


def test_normalize_payload_accepts_minimal_fields():
    payload = {"swarm_id": "s1", "replica_id": 1, "state": "running", "http_ready": True}
    out = _normalize_payload(payload)

    assert out is not None
    assert out["swarm_id"] == "s1"
    assert out["replica_id"] == 1
    assert out["http_ready"] == 1
    assert out["state"] == "running"


def test_normalize_payload_rejects_missing_ids():
    assert _normalize_payload({"replica_id": 1}) is None
    assert _normalize_payload({"swarm_id": "s1"}) is None


def test_normalize_payload_coerces_ints():
    payload = {"swarm_id": "s1", "replica_id": "2", "port": "9000", "pid": "123"}
    out = _normalize_payload(payload)

    assert out is not None
    assert out["replica_id"] == 2
    assert out["port"] == 9000
    assert out["pid"] == 123
