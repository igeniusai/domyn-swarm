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

import math
from types import SimpleNamespace

import pytest
import requests

from domyn_swarm.platform.http_probe import wait_http_200


class FakeClock:
    def __init__(self, t0: float = 0.0):
        self.t = t0
        self.sleep_calls = 0
        self.last_slept = []

    def now(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.sleep_calls += 1
        self.last_slept.append(dt)
        self.t += dt


def _http_get_sequencer(sequence):
    """
    Returns a http_get(url, timeout=...) callable that:
    - pops the next item from `sequence` each call
    - if it's an Exception instance, raises it
    - if it's an int, returns an object with .status_code set to that int
    - if sequence is exhausted, repeats the last item
    """
    seq = list(sequence)

    def _get(url, timeout=5.0, headers=None):
        # Keep verifying the function passes timeout=5.0
        assert math.isclose(timeout, 5.0)
        if seq:
            item = seq.pop(0)
        else:
            item = sequence[-1]
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(status_code=int(item))

    return _get


def test_immediate_success_no_sleep():
    clock = FakeClock()
    http_get = _http_get_sequencer([200])

    wait_http_200(
        "http://svc/ok",
        timeout_s=10,
        poll_interval_s=1.0,
        now=clock.now,
        sleep=clock.sleep,
        http_get=http_get,
    )

    assert clock.sleep_calls == 0  # returned immediately


def test_retries_then_success_sleeps_once():
    clock = FakeClock()
    http_get = _http_get_sequencer([503, 200])

    wait_http_200(
        "http://svc/retry",
        timeout_s=10,
        poll_interval_s=2.5,
        now=clock.now,
        sleep=clock.sleep,
        http_get=http_get,
    )

    assert clock.sleep_calls == 1
    assert clock.last_slept == [2.5]


def test_handles_request_exception_then_succeeds():
    clock = FakeClock()
    http_get = _http_get_sequencer([requests.RequestException("boom"), 200])

    wait_http_200(
        "http://svc/exc",
        timeout_s=10,
        poll_interval_s=1.0,
        now=clock.now,
        sleep=clock.sleep,
        http_get=http_get,
    )

    assert clock.sleep_calls == 1  # one failed attempt, one sleep, then success


def test_times_out_after_deadline():
    clock = FakeClock()
    # Always non-200 so we hit the timeout
    http_get = _http_get_sequencer([500])

    timeout_s = 10
    poll = 3
    with pytest.raises(
        RuntimeError, match="Timeout waiting for http://svc/slow to return 200 OK"
    ):
        wait_http_200(
            "http://svc/slow",
            timeout_s=timeout_s,
            poll_interval_s=poll,
            now=clock.now,
            sleep=clock.sleep,
            http_get=http_get,
        )

    # Expected number of sleeps: ceil(timeout / poll)
    assert clock.sleep_calls == math.ceil(timeout_s / poll)
    # And our fake time advanced accordingly
    assert clock.t >= timeout_s


def test_passes_url_to_http_get_and_uses_default_poll_interval():
    # We don't check sleep intervals here; just ensure the URL reaches http_get
    seen = {}

    def http_get(url, timeout=5.0, headers=None):
        seen["url"] = url
        return SimpleNamespace(status_code=200)

    # Use real time defaults but keep tiny timeout to avoid flakiness (no sleeping happens anyway)
    wait_http_200(
        "http://svc/path",
        timeout_s=1,
        http_get=http_get,
    )

    assert seen["url"] == "http://svc/path"
