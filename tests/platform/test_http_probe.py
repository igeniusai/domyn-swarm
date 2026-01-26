import types

import pytest

from domyn_swarm.platform.http_probe import HttpWaitError, get_url_status, wait_http_200


class DummyToken:
    """Token stub for settings."""

    def get_secret_value(self) -> str:
        return "secret"


def test_wait_http_200_success(monkeypatch):
    """Returns once the endpoint reports 200."""
    responses = [500, 200]

    def _http_get(url, **kwargs):
        """Fake HTTP getter that returns successive statuses."""
        status = responses.pop(0)
        return types.SimpleNamespace(status_code=status)

    def _now():
        """Deterministic clock."""
        return 0.0

    wait_http_200(
        "http://example",
        timeout_s=5,
        poll_interval_s=0.0,
        now=_now,
        sleep=lambda _: None,
        http_get=_http_get,
    )


def test_wait_http_200_timeout(monkeypatch):
    """Raises HttpWaitError when the timeout expires."""
    ticks = {"t": 0.0}

    def _now():
        """Monotonic clock that advances by 1 per call."""
        ticks["t"] += 1.0
        return ticks["t"]

    def _http_get(url, **kwargs):
        """Fake HTTP getter that always fails."""
        return types.SimpleNamespace(status_code=500)

    with pytest.raises(HttpWaitError):
        wait_http_200(
            "http://example",
            timeout_s=1,
            poll_interval_s=0.0,
            now=_now,
            sleep=lambda _: None,
            http_get=_http_get,
        )


def test_wait_http_200_includes_token(monkeypatch):
    """Adds authorization headers when a token is configured."""
    seen_headers = {}

    class DummySettings:
        """Settings stub with API token."""

        api_token = DummyToken()
        vllm_api_key = None
        singularityenv_vllm_api_key = None

    monkeypatch.setattr("domyn_swarm.platform.http_probe.get_settings", lambda: DummySettings())

    def _http_get(url, **kwargs):
        """Capture authorization headers."""
        seen_headers.update(kwargs.get("headers", {}))
        return types.SimpleNamespace(status_code=200)

    wait_http_200(
        "http://example",
        timeout_s=1,
        poll_interval_s=0.0,
        now=lambda: 0.0,
        sleep=lambda _: None,
        http_get=_http_get,
    )
    assert seen_headers["Authorization"] == "Bearer secret"


def test_get_url_status_handles_failure():
    """Returns -1 when the HTTP request fails."""

    def _http_get(url, **kwargs):
        """Fake HTTP getter that raises a RequestException."""
        import requests

        raise requests.RequestException("nope")

    assert get_url_status("http://example", http_get=_http_get) == -1
