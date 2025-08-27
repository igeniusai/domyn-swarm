from unittest.mock import MagicMock, patch

import pytest
import requests

from domyn_swarm.core.lb_health_checker import LBHealthChecker


class DummyStatus:
    def update(self, msg):
        pass


@patch("domyn_swarm.core.lb_health_checker.subprocess.check_output")
def test_lb_healthy(mock_check_output, dummy_swarm_lb_health_checker):
    dummy_swarm_lb_health_checker._get_lb_node.return_value = "dummy-node"
    checker = LBHealthChecker(dummy_swarm_lb_health_checker)

    mock_check_output.return_value = "RUNNING\n"

    # Fake HTTP and a fake clock so we don't sleep for real
    http = MagicMock()
    http.return_value.status_code = 200
    t = {"v": 0.0}

    def now():
        return t["v"]

    def fake_sleep(dt):
        t["v"] += dt

    # Inject new defaults so internal call to _wait_for_http_ready uses our fakes
    old_kw = LBHealthChecker._wait_for_http_ready.__kwdefaults__ or {}
    new_kw = {**old_kw, "http_get": http, "now": now, "sleep": fake_sleep}

    with (
        patch.object(LBHealthChecker._wait_for_http_ready, "__kwdefaults__", new_kw),
        patch("domyn_swarm.core.lb_health_checker.time.sleep", return_value=None),
    ):
        checker.wait_for_lb(1)  # must be > 0 so a probe actually runs

    assert dummy_swarm_lb_health_checker.endpoint == "http://dummy-node:8080"


def test_wait_for_jobs_to_start_raises_on_failed_state(dummy_swarm_lb_health_checker):
    checker = LBHealthChecker(dummy_swarm_lb_health_checker)
    with patch.object(checker, "_sacct_state", return_value="FAILED"):
        with pytest.raises(RuntimeError):
            checker._wait_for_jobs_to_start(DummyStatus())


def test_wait_for_jobs_to_start_raises_on_cancelled(dummy_swarm_lb_health_checker):
    checker = LBHealthChecker(dummy_swarm_lb_health_checker)
    with patch.object(checker, "_sacct_state", side_effect=["RUNNING", "CANCELLED"]):
        with pytest.raises(RuntimeError):
            checker._wait_for_jobs_to_start(DummyStatus())


def test_resolve_lb_node_sets_value(dummy_swarm_lb_health_checker):
    checker = LBHealthChecker(dummy_swarm_lb_health_checker)
    dummy_swarm_lb_health_checker._get_lb_node.return_value = "mock-node"
    with patch("domyn_swarm.core.lb_health_checker.time.sleep", return_value=None):
        checker._resolve_lb_node(DummyStatus())
    assert dummy_swarm_lb_health_checker.lb_node == "mock-node"


def test_http_ready_sets_endpoint(dummy_swarm_lb_health_checker):
    checker = LBHealthChecker(dummy_swarm_lb_health_checker)
    dummy_swarm_lb_health_checker.lb_node = "mock-node"
    dummy_swarm_lb_health_checker.cfg.lb_port = 8080
    dummy_swarm_lb_health_checker.endpoint = None

    mock_response = MagicMock()
    mock_response.status_code = 200

    # Inject http_get and a no-op sleep; use timeout > 0 to allow one probe
    checker._wait_for_http_ready(
        DummyStatus(),
        MagicMock(),
        1,
        http_get=lambda url, timeout: mock_response,
        sleep=lambda _: None,
    )

    assert dummy_swarm_lb_health_checker.endpoint == "http://mock-node:8080"


def test_http_fails_but_does_not_crash(dummy_swarm_lb_health_checker):
    checker = LBHealthChecker(dummy_swarm_lb_health_checker)
    dummy_swarm_lb_health_checker.lb_node = "mock-node"
    dummy_swarm_lb_health_checker.cfg.lb_port = 8080

    def failing_get(url, timeout):
        raise requests.RequestException("fail")

    def raising_sleep(_):
        raise RuntimeError("break loop")

    with pytest.raises(RuntimeError, match="break loop"):
        checker._wait_for_http_ready(
            DummyStatus(),
            MagicMock(),
            1,  # > 0 so loop runs and hits our raising sleep
            http_get=failing_get,
            sleep=raising_sleep,
        )
