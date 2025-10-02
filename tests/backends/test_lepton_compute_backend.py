from types import SimpleNamespace
from typing import Optional

import pytest
from leptonai.api.v1.types.deployment import (
    EnvValue,
    EnvVar,
    LeptonContainer,
    LeptonDeploymentState,
)
from leptonai.api.v1.types.job import LeptonJobUserSpec

from domyn_swarm.backends.compute.lepton import LeptonComputeBackend
from domyn_swarm.platform.protocols import JobStatus


# ------------------------------
# Test doubles for the Lepton API v2 client
# ------------------------------
class MockCreatedJob:
    """Mock object returned from client.job.create() with optional metadata.id_."""

    def __init__(self, job_id: Optional[str]):
        self.metadata = SimpleNamespace(id_=job_id)


class MockJobStatus:
    """Mock object returned from client.job.get() with job state."""

    def __init__(self, state):
        self.status = SimpleNamespace(state=state)


class MockJobAPI:
    """Mock implementation of Lepton Job API for testing."""

    def __init__(self, created_id: Optional[str] = "job-123", state=None):
        self.last_created_job = None
        self.created_id = created_id
        self.state = state
        self.deleted_job_ids = []

    def create(self, job_spec):
        self.last_created_job = job_spec
        return MockCreatedJob(self.created_id)

    def get(self, job_id: str):
        return MockJobStatus(self.state)

    def delete(self, job_id: str):
        self.deleted_job_ids.append(job_id)


class MockLeptonClient:
    """Mock Lepton client with job API."""

    def __init__(self, job_api: MockJobAPI):
        self.job = job_api


# ------------------------------
# Fixtures
# ------------------------------
@pytest.fixture
def mock_job_api():
    """Mock job API that returns successful responses by default."""
    return MockJobAPI(created_id="job-123", state=LeptonDeploymentState.Ready)


@pytest.fixture
def backend(monkeypatch, mock_job_api) -> LeptonComputeBackend:
    """Backend with mocked Lepton client."""
    backend = LeptonComputeBackend(workspace="ws-1")
    monkeypatch.setattr(backend, "_client", lambda: MockLeptonClient(mock_job_api))
    backend._job_api = mock_job_api  # expose for test assertions
    return backend


@pytest.fixture
def sample_config():
    """Sample configuration object for testing."""
    return SimpleNamespace(
        job=SimpleNamespace(
            image="igeniusai/domyn-swarm:latest",
            allowed_dedicated_node_groups=["group-a"],
            allowed_nodes=["node-1", "node-2"],
            resource_shape=None,
            mounts=None,
        ),
        endpoint=SimpleNamespace(resource_shape="gpu.4xh200"),
    )


# ------------------------------
# Job submission tests
# ------------------------------
class TestJobSubmission:
    """Tests for job submission functionality."""

    def test_submit_creates_job_with_correct_container_spec(self, backend):
        """Test that submit() creates a job with the correct container configuration."""
        handle = backend.submit(
            name="my-job",
            image="repo/image:tag",
            command=["python", "-m", "domyn_swarm.jobs.run", "--help"],
            env={"ENDPOINT": "http://x"},
            resources={
                "resource_shape": "gpu.4xh200",
                "completions": 1,
                "parallelism": 1,
            },
            extras={"api_token": "my-secret-name"},
        )

        # Verify returned handle
        assert handle.id == "job-123"
        assert handle.status is JobStatus.PENDING

        # Verify job spec sent to API
        job_spec = backend._job_api.last_created_job
        assert job_spec is not None

        # Verify container configuration
        container = job_spec.spec.container
        assert isinstance(container, LeptonContainer)
        assert container.image == "repo/image:tag"
        assert container.command == ["python", "-m", "domyn_swarm.jobs.run", "--help"]

    def test_submit_injects_api_token_as_secret_env_var(self, backend):
        """Test that submit() properly injects API token as a secret environment variable."""
        backend.submit(
            name="my-job",
            image="repo/image:tag",
            command=["python", "script.py"],
            resources={"resource_shape": "gpu.4xh200"},
            extras={"token_secret_name": "my-secret-name"},
        )

        job_spec = backend._job_api.last_created_job

        # Find API_TOKEN environment variable
        api_token_vars = [
            env
            for env in job_spec.spec.envs
            if isinstance(env, EnvVar) and env.name == "API_TOKEN"
        ]

        assert len(api_token_vars) == 1
        env_var = api_token_vars[0]
        assert isinstance(env_var.value_from, EnvValue)
        assert env_var.value_from.secret_name_ref == "my-secret-name"

    def test_submit_sets_resource_configuration(self, backend):
        """Test that submit() correctly sets resource configuration."""
        backend.submit(
            name="my-job",
            image="repo/image:tag",
            command=["python", "script.py"],
            resources={
                "resource_shape": "gpu.4xh200",
                "completions": 1,
                "parallelism": 1,
            },
            extras={"api_token": "secret"},
        )

        job_spec = backend._job_api.last_created_job
        assert isinstance(job_spec.spec, LeptonJobUserSpec)
        assert job_spec.spec.resource_shape == "gpu.4xh200"
        assert job_spec.spec.completions == 1
        assert job_spec.spec.parallelism == 1

    def test_submit_raises_error_when_job_creation_fails(self, monkeypatch):
        """Test that submit() raises RuntimeError when job creation fails."""
        backend = LeptonComputeBackend()
        failing_job_api = MockJobAPI(created_id=None)  # No ID returned = failure
        monkeypatch.setattr(
            backend, "_client", lambda: MockLeptonClient(failing_job_api)
        )

        with pytest.raises(RuntimeError, match="Failed to create Lepton job"):
            backend.submit(
                name="job",
                image="img",
                command=["python", "-c", "print(1)"],
                resources={"resource_shape": "gpu.4xh200"},
                extras={"api_token": "secret"},
            )


# ------------------------------
# Job status tests
# ------------------------------
class TestJobStatus:
    """Tests for job status checking functionality."""

    @pytest.mark.parametrize(
        "lepton_state,expected_status",
        [
            (LeptonDeploymentState.Ready, JobStatus.RUNNING),
            (LeptonDeploymentState.Stopped, JobStatus.FAILED),
            (LeptonDeploymentState.Stopping, JobStatus.FAILED),
            (LeptonDeploymentState.Starting, JobStatus.SUCCEEDED),
        ],
    )
    def test_wait_maps_lepton_states_to_job_statuses(
        self, monkeypatch, lepton_state, expected_status
    ):
        """Test that wait() correctly maps Lepton deployment states to job statuses."""
        backend = LeptonComputeBackend()
        job_api = MockJobAPI(state=lepton_state)
        monkeypatch.setattr(backend, "_client", lambda: MockLeptonClient(job_api))

        job_handle = SimpleNamespace(id="job-xyz")
        actual_status = backend.wait(job_handle)

        assert actual_status is expected_status


# ------------------------------
# Job cancellation tests
# ------------------------------
class TestJobCancellation:
    """Tests for job cancellation functionality."""

    def test_cancel_calls_delete_api(self, monkeypatch):
        """Test that cancel() calls the delete API with correct job ID."""
        backend = LeptonComputeBackend()
        job_api = MockJobAPI()
        monkeypatch.setattr(backend, "_client", lambda: MockLeptonClient(job_api))

        job_handle = SimpleNamespace(id="job-42")
        backend.cancel(job_handle)

        assert "job-42" in job_api.deleted_job_ids

    def test_cancel_handles_api_errors_gracefully(self, monkeypatch):
        """Test that cancel() doesn't raise exceptions when API call fails."""

        class FailingJobAPI(MockJobAPI):
            def delete(self, job_id: str):
                raise RuntimeError("API error")

        backend = LeptonComputeBackend()
        monkeypatch.setattr(
            backend, "_client", lambda: MockLeptonClient(FailingJobAPI())
        )

        # Should not raise an exception
        backend.cancel(SimpleNamespace(id="job-err"))


# ------------------------------
# Default configuration tests
# ------------------------------
class TestDefaultConfiguration:
    """Tests for default configuration methods."""

    def test_default_python_returns_python_executable(self, backend):
        """Test that default_python() returns the correct Python executable."""
        assert backend.default_python(cfg=None) == "python"

    def test_default_image_uses_config_image(self, sample_config):
        """Test that default_image() uses the image from configuration."""
        backend = LeptonComputeBackend()
        assert backend.default_image(sample_config) == "igeniusai/domyn-swarm:latest"

    def test_default_resources_builds_valid_spec(self, sample_config):
        """Test that default_resources() builds a valid resource specification."""
        backend = LeptonComputeBackend()
        resource_dict = backend.default_resources(sample_config)

        # Validate by creating the actual spec object
        spec = LeptonJobUserSpec.model_validate(resource_dict)
        assert spec.resource_shape == "gpu.4xh200"  # Falls back to endpoint config
        assert spec.completions == 1
        assert spec.parallelism == 1
        assert spec.affinity.allowed_dedicated_node_groups == ["group-a"]
        assert spec.affinity.allowed_nodes_in_node_group == ["node-1", "node-2"]
