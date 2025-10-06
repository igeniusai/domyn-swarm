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

from types import SimpleNamespace

import pytest

from domyn_swarm.backends.serving.lepton import LeptonServingBackend
from domyn_swarm.platform.protocols import ServingHandle, ServingPhase

# ------------------------------
# Test Doubles for Lepton SDK
# ------------------------------


class _ModelLike(SimpleNamespace):
    """SimpleNamespace with a pydantic-like .model_dump()"""

    def model_dump(self, **_: dict) -> dict:
        def _dump(obj):
            if isinstance(obj, _ModelLike):
                return {k: _dump(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [_dump(v) for v in obj]
            elif isinstance(obj, dict):
                return {k: _dump(v) for k, v in obj.items()}
            else:
                return obj

        return _dump(self)


class FakeDeploymentAPI:
    """Mock for Lepton deployment API with configurable behavior."""

    def __init__(
        self,
        create_succeeds: bool = True,
        ready_after_calls: int = 0,
        endpoint_url: str = "http://example.com",
    ):
        self.create_succeeds = create_succeeds
        self.ready_after_calls = ready_after_calls
        self.endpoint_url = endpoint_url
        self.get_call_count = 0
        self.deleted_deployment_ids = []
        self.last_created_deployment = None

    def create(self, deployment_obj):
        self.last_created_deployment = deployment_obj
        return self.create_succeeds

    def get(self, deployment_name: str):
        self.get_call_count += 1

        if self.get_call_count > self.ready_after_calls:
            return self._create_ready_deployment()
        else:
            return self._create_creating_deployment()

    def delete(self, deployment_id: str):
        self.deleted_deployment_ids.append(deployment_id)

    def _create_ready_deployment(self):
        endpoint = _ModelLike(external_endpoint=self.endpoint_url)
        status = _ModelLike(state="Ready", endpoint=endpoint)
        # Top-level object mirrors what SDK returns (e.g., dep.status.state / .endpoint)
        return _ModelLike(status=status)

    def _create_creating_deployment(self):
        status = _ModelLike(state="Creating", endpoint=None)
        return _ModelLike(status=status)


class FakeSecretAPI:
    """Mock for Lepton secret API."""

    def __init__(self):
        self.created_secrets = None
        self.deleted_secret_names = []

    def create(self, secrets):
        self.created_secrets = secrets
        return True

    def delete(self, secret_name: str):
        self.deleted_secret_names.append(secret_name)


class FakeLeptonClient:
    """Mock Lepton SDK client."""

    def __init__(self, deployment_api: FakeDeploymentAPI, secret_api: FakeSecretAPI):
        self.deployment = deployment_api
        self.secret = secret_api


# ------------------------------
# Test Fixtures
# ------------------------------


@pytest.fixture
def mock_backend(monkeypatch) -> LeptonServingBackend:
    """Create a LeptonServingBackend with mocked dependencies."""
    backend = LeptonServingBackend(workspace="test-workspace")

    # Set up mocks
    deployment_api = FakeDeploymentAPI(
        create_succeeds=True, ready_after_calls=0, endpoint_url="http://test.com"
    )
    secret_api = FakeSecretAPI()
    fake_client = FakeLeptonClient(deployment_api, secret_api)

    # Apply patches
    monkeypatch.setattr(backend, "_client", lambda: fake_client)

    import domyn_swarm.backends.serving.lepton as lepton_module

    monkeypatch.setattr(
        lepton_module, "get_env_var_by_name", lambda envs, name: "test-secret"
    )
    monkeypatch.setattr(
        lepton_module,
        "sanitize_tokens_in_deployment",
        lambda deployed: _ModelLike(raw="SANITIZED"),
    )

    # Expose mocks for test assertions
    backend._deployment_api = deployment_api
    backend._secret_api = secret_api
    return backend


def create_test_spec(**overrides):
    """Helper to create deployment specs with sensible defaults."""
    default_spec = {
        "resource_shape": "gpu.4xh200",
        "replicas": 1,
        "envs": [{"name": "FOO", "value": "BAR"}],
        "api_tokens": [{"value": "test-token-value"}],
    }
    return {**default_spec, **overrides}


def extract_secret_info(secret_item):
    """Extract name and value from a secret item (handles both object and dict)."""
    name = getattr(secret_item, "name", None) or secret_item.get("name")
    value = getattr(secret_item, "value", None) or secret_item.get("value")
    return name, value


# ------------------------------
# Tests for create_or_update
# ------------------------------


def test_create_or_update_creates_serving_handle_with_correct_attributes(mock_backend):
    """Test that create_or_update returns a properly configured ServingHandle."""
    spec = create_test_spec()

    handle = mock_backend.create_or_update("test-endpoint", spec, None)

    print(handle)
    assert isinstance(handle, ServingHandle)
    assert handle.id == "test-endpoint"
    assert handle.url == "http://test.com"
    assert handle.meta["name"] == "test-endpoint"
    assert handle.meta["raw"]["raw"] == "SANITIZED"
    assert handle.meta["token_secret_name"] == "test-secret"


def test_create_or_update_creates_secret_with_token_from_spec(mock_backend):
    """Test that create_or_update creates a secret using the API token from the spec."""
    spec = create_test_spec(api_tokens=[{"value": "my-api-token"}])

    mock_backend.create_or_update("test-endpoint", spec, None)

    created_secrets = mock_backend._secret_api.created_secrets
    assert created_secrets is not None
    assert len(created_secrets) == 1

    secret_name, secret_value = extract_secret_info(created_secrets[0])
    assert secret_name == "test-secret"
    assert secret_value == "my-api-token"


def test_create_or_update_uses_defaults_when_env_and_token_missing(monkeypatch):
    """Test fallback behavior when environment variables and API tokens are not provided."""
    backend = LeptonServingBackend()
    deployment_api = FakeDeploymentAPI(endpoint_url="http://fallback.com")
    secret_api = FakeSecretAPI()

    monkeypatch.setattr(
        backend, "_client", lambda: FakeLeptonClient(deployment_api, secret_api)
    )

    import domyn_swarm.backends.serving.lepton as lepton_module

    monkeypatch.setattr(lepton_module, "get_env_var_by_name", lambda envs, name: None)
    monkeypatch.setattr(
        lepton_module, "sanitize_tokens_in_deployment", lambda deployed: deployed
    )

    spec = create_test_spec(envs=[], api_tokens=[])

    _ = backend.create_or_update("test-endpoint", spec, None)

    secret_name, secret_value = extract_secret_info(secret_api.created_secrets[0])
    assert secret_name == "test-endpoint-token"
    assert secret_value == "changeme"


def test_create_or_update_raises_runtime_error_when_deployment_creation_fails(
    monkeypatch,
):
    """Test that create_or_update raises RuntimeError when deployment creation fails."""
    backend = LeptonServingBackend()
    failing_deployment_api = FakeDeploymentAPI(create_succeeds=False)
    secret_api = FakeSecretAPI()

    monkeypatch.setattr(
        backend, "_client", lambda: FakeLeptonClient(failing_deployment_api, secret_api)
    )

    with pytest.raises(RuntimeError, match="Failed to create Lepton deployment"):
        backend.create_or_update("test-endpoint", create_test_spec(), None)


# ------------------------------
# Tests for wait_ready
# ------------------------------


def test_wait_ready_succeeds_when_deployment_becomes_ready(monkeypatch):
    """Test that wait_ready succeeds when deployment transitions to ready state."""
    backend = LeptonServingBackend()
    deployment_api = FakeDeploymentAPI(
        ready_after_calls=2, endpoint_url="http://ready.com"
    )
    secret_api = FakeSecretAPI()

    monkeypatch.setattr(
        backend, "_client", lambda: FakeLeptonClient(deployment_api, secret_api)
    )
    _mock_time_functions(monkeypatch)

    handle = ServingHandle(id="test-endpoint", url="", meta={"name": "test-endpoint"})
    result = backend.wait_ready(handle, 10, None)

    assert result is handle


def test_wait_ready_raises_timeout_error_when_deployment_never_ready(monkeypatch):
    """Test that wait_ready raises TimeoutError when deployment doesn't become ready in time."""
    backend = LeptonServingBackend()
    deployment_api = FakeDeploymentAPI(ready_after_calls=10_000)  # Never ready
    secret_api = FakeSecretAPI()

    monkeypatch.setattr(
        backend, "_client", lambda: FakeLeptonClient(deployment_api, secret_api)
    )
    _mock_time_to_advance_quickly(monkeypatch)

    handle = ServingHandle(id="test-endpoint", url="", meta={"name": "test-endpoint"})

    with pytest.raises(TimeoutError, match="Timed out waiting for Lepton deployment"):
        backend.wait_ready(handle, 5, None)


def _mock_time_functions(monkeypatch):
    """Mock time functions to avoid real delays during testing."""
    advancing_time = [0, 1, 2, 3, 4, 5]

    def fake_time():
        return advancing_time.pop(0) if advancing_time else 100

    monkeypatch.setattr("time.time", fake_time)
    monkeypatch.setattr("time.sleep", lambda seconds: None)


def _mock_time_to_advance_quickly(monkeypatch):
    """Mock time to advance quickly past timeout."""
    current_time = {"value": 0}

    def fake_time():
        current_time["value"] += 6  # Jump past sleep interval
        return current_time["value"]

    monkeypatch.setattr("time.time", fake_time)
    monkeypatch.setattr("time.sleep", lambda seconds: None)


# ------------------------------
# Tests for delete
# ------------------------------


def test_delete_removes_both_deployment_and_secret(monkeypatch):
    """Test that delete removes both the deployment and its associated secret."""
    backend = LeptonServingBackend()
    deployment_api = FakeDeploymentAPI()
    secret_api = FakeSecretAPI()

    monkeypatch.setattr(
        backend, "_client", lambda: FakeLeptonClient(deployment_api, secret_api)
    )

    handle = ServingHandle(
        id="test-endpoint",
        url="http://test.com",
        meta={"token_secret_name": "test-secret"},
    )

    backend.delete(handle)

    assert "test-endpoint" in deployment_api.deleted_deployment_ids
    assert "test-secret" in secret_api.deleted_secret_names


def test_delete_handles_api_errors_gracefully(monkeypatch):
    """Test that delete doesn't raise exceptions when API calls fail."""

    class FailingDeploymentAPI(FakeDeploymentAPI):
        def delete(self, deployment_id: str):
            raise RuntimeError("Deployment deletion failed")

    class FailingSecretAPI(FakeSecretAPI):
        def delete(self, secret_name: str):
            raise RuntimeError("Secret deletion failed")

    backend = LeptonServingBackend()
    monkeypatch.setattr(
        backend,
        "_client",
        lambda: FakeLeptonClient(FailingDeploymentAPI(), FailingSecretAPI()),
    )

    handle = ServingHandle(
        id="test-endpoint", url="", meta={"token_secret_name": "test-secret"}
    )

    # Should not raise any exceptions
    backend.delete(handle)


# ------------------------------
# Tests for ensure_ready
# ------------------------------


def test_ensure_ready_raises_error_when_handle_has_no_url():
    """Test that ensure_ready raises RuntimeError when ServingHandle has no URL."""
    backend = LeptonServingBackend()
    handle = ServingHandle(id="test-endpoint", url="", meta={})

    with pytest.raises(RuntimeError, match="Swarm not ready"):
        backend.ensure_ready(handle)


def test_ensure_ready_succeeds_when_handle_has_url():
    """Test that ensure_ready passes when ServingHandle has a valid URL."""
    backend = LeptonServingBackend()
    handle = ServingHandle(id="test-endpoint", url="http://ready.com", meta={})

    # Should not raise any exceptions
    backend.ensure_ready(handle)


# ------------------------------
# Tests for status
# ------------------------------
def _handle(name="ep1", url=""):
    return ServingHandle(id=name, url=url, meta={"name": name})


def test_status_running_when_ready_and_http_ok(mock_backend, monkeypatch, mocker):
    # Fake client already returns state="Ready" and url="http://test.com" via mock_backend fixture

    # Patch requests.get to return 200 OK
    import domyn_swarm.backends.serving.lepton as lepton_module

    get_mock = mocker.Mock(return_value=SimpleNamespace(status_code=200))
    monkeypatch.setattr(lepton_module, "requests", SimpleNamespace(get=get_mock))

    h = _handle()
    st = mock_backend.status(h)

    # One HTTP probe to /v1/models with a small timeout
    get_mock.assert_called_once()
    assert get_mock.call_args.args[0] == "http://test.com/v1/models"
    assert get_mock.call_args.kwargs.get("timeout") in (1.0, 1.5, 2.0)

    # Phase and URL cached on handle
    assert st.phase == ServingPhase.RUNNING
    assert h.url == "http://test.com"

    # Deployment.get called exactly once
    assert mock_backend._deployment_api.get_call_count == 1


def test_status_initializing_when_ready_but_http_down(
    mock_backend, monkeypatch, mocker
):
    # Ready at the platform level, but HTTP probe fails
    import requests

    import domyn_swarm.backends.serving.lepton as lepton_module

    get_mock = mocker.Mock(side_effect=requests.RequestException("boom"))
    monkeypatch.setattr(lepton_module, "requests", SimpleNamespace(get=get_mock))

    h = _handle()
    st = mock_backend.status(h)

    assert st.phase == ServingPhase.INITIALIZING
    # URL not updated on failed probe
    assert h.url in ("", None)
    get_mock.assert_called_once()
    assert mock_backend._deployment_api.get_call_count == 1


def test_status_pending_when_creating_state(mock_backend, monkeypatch, mocker):
    # Force the FakeDeploymentAPI to report "Creating" (transitional) on first call
    mock_backend._deployment_api.ready_after_calls = (
        100  # first get() returns "Creating"
    )

    # Even if HTTP would be OK, status() should stop at PENDING for non-Ready state
    import domyn_swarm.backends.serving.lepton as lepton_module

    get_mock = mocker.Mock(return_value=SimpleNamespace(status_code=200))
    monkeypatch.setattr(lepton_module, "requests", SimpleNamespace(get=get_mock))

    h = _handle()
    st = mock_backend.status(h)

    assert st.phase == ServingPhase.PENDING
    # No HTTP probe should be required for non-Ready, but
    # if your implementation still probes, allow 0 or 1.
    assert get_mock.call_count in (0, 1)
    assert mock_backend._deployment_api.get_call_count == 1
    assert h.url in ("", None)


def test_status_failed_on_stopped_state(mock_backend, monkeypatch):
    # Make the deployment API return Stopped
    def _stopped(_name: str):
        status = SimpleNamespace(state="Stopped", endpoint=None)
        return SimpleNamespace(status=status)

    mock_backend._deployment_api.get_call_count = 0
    monkeypatch.setattr(mock_backend._deployment_api, "get", _stopped)

    h = _handle()
    st = mock_backend.status(h)

    assert st.phase == ServingPhase.FAILED
    assert (
        mock_backend._deployment_api.get_call_count == 0
        or mock_backend._deployment_api.get_call_count == 1
    )
    assert h.url in ("", None)


def test_status_unknown_when_client_raises(mock_backend, monkeypatch):
    # Make _client() raise to simulate Lepton API outage
    monkeypatch.setattr(
        mock_backend, "_client", lambda: (_ for _ in ()).throw(Exception("api down"))
    )

    h = _handle()
    st = mock_backend.status(h)

    assert st.phase == ServingPhase.UNKNOWN
    # No URL change
    assert h.url in ("", None)
