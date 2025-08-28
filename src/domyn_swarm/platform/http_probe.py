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
