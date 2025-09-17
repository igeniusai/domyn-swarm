import types
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
class _DummyCreated:
    """Mimic the returned object from client.job.create(...) with metadata.id_ present or None."""

    class _Meta:
        def __init__(self, id_: Optional[str]):
            self.id_ = id_

    def __init__(self, id_: Optional[str]):
        # Always present; id_ may be None to simulate failure
        self.metadata = _DummyCreated._Meta(id_)


class _DummyJobObj:
    """Mimic client.job.get(..) result with .status.state."""

    class _Status:
        def __init__(self, state):
            self.state = state

    def __init__(self, state):
        self.status = _DummyJobObj._Status(state)


class _FakeJobAPI:
    def __init__(self, created_id: Optional[str] = "job-123", state=None):
        self._last_created_job = None
        self._created_id = created_id
        self._state = state
        self._deleted_ids = []

    def create(self, job_obj):
        self._last_created_job = job_obj
        # Always return an object that has .metadata; id_ may be None
        return _DummyCreated(self._created_id)

    def get(self, job_id: str):
        return _DummyJobObj(self._state)

    def delete(self, job_id: str):
        self._deleted_ids.append(job_id)


class _FakeClient:
    def __init__(self, job_api: _FakeJobAPI):
        self.job = job_api


# ------------------------------
# Fixtures
# ------------------------------
@pytest.fixture
def backend(monkeypatch) -> LeptonComputeBackend:
    """Backend with _client patched to our fake client."""
    be = LeptonComputeBackend(workspace="ws-1")
    # default fake job api returns id 'job-123' and state Ready
    job_api = _FakeJobAPI(created_id="job-123", state=LeptonDeploymentState.Ready)
    monkeypatch.setattr(be, "_client", lambda: _FakeClient(job_api))
    be._job_api = job_api  # expose for assertions
    return be


@pytest.fixture
def minimal_cfg():
    """
    Minimal config-like object with .job and .endpoint fields.
    We avoid importing the real LeptonConfig; the backend only reads attributes.
    """
    job = SimpleNamespace(
        image="igeniusai/domyn-swarm:latest",
        allowed_dedicated_node_groups=["group-a"],
        allowed_nodes=["node-1", "node-2"],
        resource_shape=None,
        mounts=None,
    )
    endpoint = SimpleNamespace(resource_shape="gpu.4xh200")
    return SimpleNamespace(job=job, endpoint=endpoint)


# ------------------------------
# submit()
# ------------------------------
def test_submit_builds_container_and_injects_secret_env(backend: LeptonComputeBackend):
    handle = backend.submit(
        name="my-job",
        image="repo/image:tag",
        command=["python", "-m", "domyn_swarm.jobs.run", "--help"],
        env={"ENDPOINT": "http://x"},  # (ignored by backend; SDK spec holds envs)
        resources={"resource_shape": "gpu.4xh200", "completions": 1, "parallelism": 1},
        extras={"api_token": "my-secret-name"},
    )

    # Returned handle
    assert handle.id == "job-123"
    assert handle.status is JobStatus.PENDING

    # What we sent to the client:
    sent = backend._job_api._last_created_job
    assert sent is not None, "client.job.create should have been called"

    # Container populated
    assert isinstance(sent.spec.container, LeptonContainer)
    assert sent.spec.container.image == "repo/image:tag"
    assert sent.spec.container.command == [
        "python",
        "-m",
        "domyn_swarm.jobs.run",
        "--help",
    ]

    # Env contains a secret ref for API_TOKEN
    assert isinstance(sent.spec.envs, list)
    api_token_envs = [
        e for e in sent.spec.envs if isinstance(e, EnvVar) and e.name == "API_TOKEN"
    ]
    assert len(api_token_envs) == 1
    ev = api_token_envs[0]
    assert isinstance(ev.value_from, EnvValue)
    assert ev.value_from.secret_name_ref == "my-secret-name"

    # Resource shape passed through
    # (sent.spec is a LeptonJobUserSpec instance)
    assert isinstance(sent.spec, LeptonJobUserSpec)
    assert sent.spec.resource_shape == "gpu.4xh200"
    assert sent.spec.completions == 1
    assert sent.spec.parallelism == 1


def test_submit_raises_if_api_returns_no_id(monkeypatch):
    be = LeptonComputeBackend()
    job_api = _FakeJobAPI(created_id=None)  # simulate missing metadata.id_
    monkeypatch.setattr(be, "_client", lambda: _FakeClient(job_api))

    with pytest.raises(RuntimeError, match="Failed to create Lepton job"):
        be.submit(
            name="job",
            image="img",
            command=["python", "-c", "print(1)"],
            resources={"resource_shape": "gpu.4xh200"},
            extras={"api_token": "sec"},
        )


# ------------------------------
# wait()
# ------------------------------
@pytest.mark.parametrize(
    "state,expected",
    [
        (LeptonDeploymentState.Ready, JobStatus.RUNNING),
        (LeptonDeploymentState.Stopped, JobStatus.FAILED),
        (LeptonDeploymentState.Stopping, JobStatus.FAILED),
        (LeptonDeploymentState.Starting, JobStatus.SUCCEEDED),
    ],
)
def test_wait_state_mapping(monkeypatch, state, expected):
    be = LeptonComputeBackend()
    job_api = _FakeJobAPI(state=state)
    monkeypatch.setattr(be, "_client", lambda: _FakeClient(job_api))

    # Make a minimal handle-like object
    handle = types.SimpleNamespace(id="job-xyz")
    assert be.wait(handle) is expected


# ------------------------------
# cancel()
# ------------------------------
def test_cancel_calls_delete(monkeypatch):
    be = LeptonComputeBackend()
    job_api = _FakeJobAPI()
    monkeypatch.setattr(be, "_client", lambda: _FakeClient(job_api))

    handle = types.SimpleNamespace(id="job-42")
    be.cancel(handle)
    assert "job-42" in job_api._deleted_ids


def test_cancel_swallows_errors(monkeypatch):
    class _ErrJobAPI(_FakeJobAPI):
        def delete(self, job_id: str):
            raise RuntimeError("boom")

    be = LeptonComputeBackend()
    monkeypatch.setattr(be, "_client", lambda: _FakeClient(_ErrJobAPI()))
    be.cancel(types.SimpleNamespace(id="job-err"))  # should not raise


# ------------------------------
# defaults
# ------------------------------
def test_default_python_returns_container_python(backend: LeptonComputeBackend):
    assert backend.default_python(cfg=None) == "python"


def test_default_image_uses_cfg_image(minimal_cfg):
    be = LeptonComputeBackend()
    assert be.default_image(minimal_cfg) == "igeniusai/domyn-swarm:latest"


def test_default_resources_builds_spec_dict(minimal_cfg):
    be = LeptonComputeBackend()
    d = be.default_resources(minimal_cfg)
    # Round-trip into model to assert fields
    spec = LeptonJobUserSpec.model_validate(d)
    assert (
        spec.resource_shape == "gpu.4xh200"
    )  # cfg.job.resource_shape was None → falls back to endpoint
    assert spec.completions == 1
    assert spec.parallelism == 1
    assert spec.affinity.allowed_dedicated_node_groups == ["group-a"]
    assert spec.affinity.allowed_nodes_in_node_group == ["node-1", "node-2"]
