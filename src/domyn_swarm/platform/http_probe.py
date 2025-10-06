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

import time
from typing import Callable

import requests


def wait_http_200(
    url: str,
    *,
    timeout_s: int,
    poll_interval_s: float = 5.0,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    http_get: Callable = requests.get,
) -> None:
    deadline = now() + timeout_s
    while now() < deadline:
        try:
            r = http_get(url, timeout=5.0)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        sleep(poll_interval_s)
    raise RuntimeError(f"Timeout waiting for {url} to return 200 OK")


def get_url_status(url: str, http_get: Callable = requests.get) -> int:
    try:
        r = http_get(url, timeout=5.0)
        return r.status_code
    except requests.RequestException:
        return -1  # Indicate failure to reach the URL
