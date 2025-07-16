from unittest.mock import patch, MagicMock
import pytest
import requests
from domyn_swarm.core.lb_health_checker import LBHealthChecker


class DummyStatus:
    def update(self, msg):
        pass


@patch("domyn_swarm.core.lb_health_checker.requests.get")
@patch("domyn_swarm.core.lb_health_checker.subprocess.check_output")
def test_lb_healthy(mock_check_output, mock_requests, dummy_swarm_lb_health_checker):
    dummy_swarm_lb_health_checker._get_lb_node.return_value = "dummy-node"
    checker = LBHealthChecker(dummy_swarm_lb_health_checker)

    mock_check_output.return_value = "RUNNING\n"
    mock_requests.return_value.status_code = 200

    # patch sleep to speed up test
    with patch("domyn_swarm.core.lb_health_checker.time.sleep"):
        checker.wait_for_lb()

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
    with patch(
        "domyn_swarm.core.lb_health_checker.requests.get", return_value=mock_response
    ):
        checker._wait_for_http_ready(DummyStatus(), MagicMock())

    assert dummy_swarm_lb_health_checker.endpoint == "http://mock-node:8080"


def test_http_fails_but_does_not_crash(dummy_swarm_lb_health_checker):
    checker = LBHealthChecker(dummy_swarm_lb_health_checker)
    dummy_swarm_lb_health_checker.lb_node = "mock-node"
    dummy_swarm_lb_health_checker.cfg.lb_port = 8080

    with (
        patch(
            "domyn_swarm.core.lb_health_checker.requests.get",
            side_effect=requests.RequestException("fail"),
        ),
        patch(
            "domyn_swarm.core.lb_health_checker.time.sleep",
            side_effect=RuntimeError("break loop"),
        ),
    ):
        with pytest.raises(RuntimeError, match="break loop"):
            checker._wait_for_http_ready(DummyStatus(), MagicMock())
