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

import pytest

# SUT
from domyn_swarm.backends.serving.slurm_readiness import SlurmReadiness
from domyn_swarm.platform.protocols import ServingHandle

# ---------- Fakes ----------


class FakeDriver:
    """Fake SlurmDriver with scripted job states and node lookup."""

    def __init__(self, rep_states, lb_states, node="nodeA"):
        self.rep_states = list(rep_states)
        self.lb_states = list(lb_states)
        self.rep_calls = 0
        self.lb_calls = 0
        self.node = node
        self.get_node_calls = 0

    def get_job_state(self, jobid: int) -> str:
        # Return next state; stick to last once exhausted
        self.rep_calls += 1
        if jobid == 111:  # replica array id in tests
            seq = self.rep_states
        else:  # lb job id in tests
            seq = self.lb_states
            self.lb_calls += 0  # keep attribute around; not essential

        idx = self.rep_calls - 1 if jobid == 111 else self.lb_calls - 1  # not used, but harmless

        if jobid == 111:
            return seq[min(self.rep_calls - 1, len(seq) - 1)]
        else:
            # Count lb calls independently
            idx = getattr(self, "_lb_idx", 0)
            state = seq[min(idx, len(seq) - 1)]
            self._lb_idx = idx + 1
            return state

    def get_node_from_jobid(self, jobid: int) -> str:
        self.get_node_calls += 1
        return self.node


class DummyStatusCtx:
    """Context manager returned by console.status(). Provides .update()."""

    def __init__(self):
        self.messages = []

    def __enter__(self):
        return self  # acts as Status

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, message: str):
        self.messages.append(message)


class DummyConsole:
    def __init__(self):
        self.status_calls = []
        self.print_calls = []

    def status(self, message: str):
        self.status_calls.append(message)
        return DummyStatusCtx()

    def print(self, *args, **kwargs):
        self.print_calls.append((args, kwargs))


# ---------- Helpers ----------


def patch_sleep_fast(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)


# ---------- Tests ----------


def test_wait_ready_happy_path(monkeypatch):
    # Replica and LB move PENDING -> RUNNING; lb_node resolved by driver
    driver = FakeDriver(
        rep_states=["PENDING", "RUNNING"], lb_states=["PENDING", "RUNNING"], node="n01"
    )
    console = DummyConsole()
    readiness = SlurmReadiness(
        driver=driver, endpoint_port=9000, poll_interval_s=0.01, console=console
    )

    # Patch HTTP probe to capture the URL
    called = {}
    import domyn_swarm.backends.serving.slurm_readiness as mod

    def fake_probe(url, timeout_s, poll_interval_s):
        called["url"] = url
        called["timeout_s"] = timeout_s
        called["poll"] = poll_interval_s
        return None

    monkeypatch.setattr(mod, "wait_http_200", fake_probe)

    patch_sleep_fast(monkeypatch)

    handle = ServingHandle(id="ep1", url="", meta={"jobid": 111, "lb_jobid": 222})
    out = readiness.wait_ready(handle, timeout_s=123)

    assert out is handle
    assert handle.meta["lb_node"] == "n01"
    assert handle.url == "http://n01:9000"
    assert called["url"] == "http://n01:9000/health"
    assert called["timeout_s"] == 123


def test_wait_ready_uses_existing_lb_node_no_driver_lookup(monkeypatch):
    driver = FakeDriver(rep_states=["RUNNING"], lb_states=["RUNNING"], node="should-not-be-used")
    console = DummyConsole()
    readiness = SlurmReadiness(
        driver=driver, endpoint_port=8000, poll_interval_s=0.01, console=console
    )

    import domyn_swarm.backends.serving.slurm_readiness as mod

    monkeypatch.setattr(mod, "wait_http_200", lambda *a, **k: None)
    patch_sleep_fast(monkeypatch)

    handle = ServingHandle(id="ep2", url="", meta={"jobid": 111, "lb_jobid": 222, "lb_node": "nZZ"})
    out = readiness.wait_ready(handle, timeout_s=50)

    assert out.url == "http://nZZ:8000"
    assert driver.get_node_calls == 0  # did not call driver.get_node_from_jobid


@pytest.mark.parametrize("missing", ["jobid", "lb_jobid"])
def test_wait_ready_raises_when_ids_missing(missing):
    driver = FakeDriver(rep_states=["RUNNING"], lb_states=["RUNNING"])
    readiness = SlurmReadiness(driver=driver, endpoint_port=9000)
    meta = {"jobid": 111, "lb_jobid": 222}
    del meta[missing]
    handle = ServingHandle(id="ep", url="", meta=meta)

    with pytest.raises(RuntimeError, match="requires 'jobid' and 'lb_jobid'"):
        readiness.wait_ready(handle, timeout_s=10)


def test_wait_jobs_running_raises_on_bad_rep_state(monkeypatch):
    driver = FakeDriver(rep_states=["FAILED"], lb_states=["RUNNING"])
    readiness = SlurmReadiness(driver=driver, endpoint_port=9000)
    patch_sleep_fast(monkeypatch)

    with pytest.raises(RuntimeError, match="Replica array ended in FAILED"):
        readiness._wait_jobs_running(111, 222, DummyStatusCtx())


def test_wait_jobs_running_raises_on_bad_lb_state(monkeypatch):
    driver = FakeDriver(rep_states=["RUNNING"], lb_states=["TIMEOUT"])
    readiness = SlurmReadiness(driver=driver, endpoint_port=9000)
    patch_sleep_fast(monkeypatch)

    with pytest.raises(RuntimeError, match="LB job ended in TIMEOUT"):
        readiness._wait_jobs_running(111, 222, DummyStatusCtx())


def test_wait_jobs_running_handles_unknown_then_pending_then_running(monkeypatch):
    # Sequence: UNKNOWN -> PENDING -> RUNNING for replicas, LB mirrors it
    driver = FakeDriver(
        rep_states=["UNKNOWN", "PENDING", "RUNNING"],
        lb_states=["UNKNOWN", "PENDING", "RUNNING"],
    )
    readiness = SlurmReadiness(driver=driver, endpoint_port=9000, poll_interval_s=0.001)
    patch_sleep_fast(monkeypatch)

    # Should return without raising
    readiness._wait_jobs_running(111, 222, DummyStatusCtx())
