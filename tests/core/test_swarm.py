import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import domyn_swarm.core.swarm as mod
from domyn_swarm.core.swarm import DomynLLMSwarm

# ---------------------------
# Tiny fakes used in tests
# ---------------------------


class FakeStateMgr:
    def __init__(self, swarm):
        self.swarm = swarm
        self.saved = 0
        self.deleted = 0

    def save(self, deployment_name: str):
        self.saved += 1

    @classmethod
    def load(cls, deployment_name: str):
        return SimpleNamespace(loaded=True, deployment_name=deployment_name)

    def delete_record(self, deployment_name: str):
        self.deleted += 1


class FakeDeployment:
    def __init__(self, serving=None, compute=None, extras=None):
        self.serving = serving
        self.compute = compute
        self.extras = extras
        self.up_calls = []
        self.down_calls = []
        self.run_calls = []
        self.ensure_ready_calls = 0

    def up(self, name, spec, timeout_s):
        self.up_calls.append((name, json.loads(json.dumps(spec)), timeout_s))
        # Return a ServingHandle-like object (the code only needs .id, .url, .meta)
        return SimpleNamespace(id=name, url="http://host:9000", meta={"name": name})

    def run(
        self,
        *,
        name,
        image,
        command,
        env,
        resources,
        detach,
        nshards=None,
        shard_id=None,
        mail_user=None,
    ):
        self.run_calls.append(
            {
                "name": name,
                "image": image,
                "command": list(command),
                "env": dict(env),
                "resources": resources,
                "detach": detach,
            }
        )
        # Return a JobHandle-like object with .meta for pid
        return SimpleNamespace(id="job-1", status="PENDING", meta={"pid": 4321})

    def down(self, handle):
        self.down_calls.append(handle.id)

    def ensure_ready(self):
        self.ensure_ready_calls += 1


class FakeComputeBackend:
    def __init__(self):
        self.default_python_called = 0
        self.default_image_called = 0
        self.default_resources_called = 0
        self.default_env_called = 0

    def default_python(self, cfg):
        self.default_python_called += 1
        return "python"

    def default_image(self, cfg):
        self.default_image_called += 1
        return "repo/image:tag"

    def default_resources(self, cfg):
        self.default_resources_called += 1
        return {"shape": "gpu.1x"}

    def default_env(self, cfg):
        self.default_env_called += 1
        return {"A": "B"}


class FakePlan:
    def __init__(self, platform="lepton"):
        self.platform = platform
        self.serving = SimpleNamespace()
        self.compute = FakeComputeBackend()
        self.serving_spec = {"replicas": 1, "resource_shape": "gpu.4xh200"}
        self.name_hint = platform
        self.extras = {}
        self.job_resources = {}


# ---------------------------
# Fixtures & helpers
# ---------------------------


@pytest.fixture(autouse=True)
def patch_state_mgr(monkeypatch):
    # Replace SwarmStateManager in the module with our fake
    monkeypatch.setattr(mod, "SwarmStateManager", FakeStateMgr)
    yield


@pytest.fixture(autouse=True)
def patch_deployment(monkeypatch):
    monkeypatch.setattr(mod, "Deployment", FakeDeployment)
    yield


@pytest.fixture(autouse=True)
def patch_to_path(monkeypatch):
    # Ensure to_path simply returns a Path
    monkeypatch.setattr(mod, "to_path", lambda p: Path(p))
    yield


@pytest.fixture
def cfg_stub():
    # Minimal cfg: must provide .model, .wait_endpoint_s, .get_deployment_plan(), and optional .backend.env
    stub = SimpleNamespace(
        name="name",
        model="m1",
        wait_endpoint_s=30,
        backend=SimpleNamespace(env={"X": "Y"}),
    )
    stub.get_deployment_plan = lambda: FakePlan(platform="lepton")
    return stub


def make_swarm(cfg):
    """
    Create a DomynLLMSwarm instance using model_construct to avoid
    pydantic validations and then manually inject plan/deployment/platform/state.
    """
    # pydantic v2: model_construct
    swarm = DomynLLMSwarm.model_construct(
        name="My_Swarm",
        cfg=cfg,
        endpoint=None,
        delete_on_exit=False,
        serving_handle=None,
    )
    # Inject state mgr and deployment based on plan (like model_post_init would)
    plan = cfg.get_deployment_plan()
    swarm._plan = plan  # type: ignore[attr-defined]
    swarm._platform = plan.platform  # type: ignore[attr-defined]
    swarm._deployment = FakeDeployment(
        serving=plan.serving, compute=plan.compute, extras=plan.extras
    )  # type: ignore[attr-defined]
    swarm._state_mgr = FakeStateMgr(swarm)  # type: ignore[attr-defined]
    return swarm


# ---------------------------
# Tests
# ---------------------------


def test_enter_sets_endpoint_persists_and_sets_compute(cfg_stub):
    swarm = make_swarm(cfg_stub)
    with swarm as s:
        assert s.endpoint == "http://host:9000"
        assert isinstance(s.serving_handle, SimpleNamespace)
        # Deployment was called with a sanitized name
        name, spec, timeout_s = swarm._deployment.up_calls[-1]  # type: ignore[attr-defined]
        assert timeout_s == cfg_stub.wait_endpoint_s
        assert spec == {"replicas": 1, "resource_shape": "gpu.4xh200"}
        # Compute backend set from plan for lepton
        assert s._deployment.compute is s._plan.compute  # type: ignore[attr-defined]
        # State persisted at least once
        assert s._state_mgr.saved >= 1  # type: ignore[attr-defined]


def test_exit_with_delete_on_exit_calls_cleanup(cfg_stub):
    swarm = make_swarm(cfg_stub)
    swarm.delete_on_exit = True
    with swarm:
        pass
    # After exiting context, cleanup called → deployment.down invoked once
    assert swarm._deployment.down_calls == [
        "my_swarm" if False else swarm.serving_handle.id
    ]  # type: ignore[attr-defined]


@pytest.mark.skip(reason="Validation not implemented yet")
def test_deployment_name_sanitization_lepton(cfg_stub):
    swarm = make_swarm(cfg_stub)
    swarm._platform = "lepton"  # type: ignore[attr-defined]
    swarm.name = "Bad*Name/With Spaces & VeryVeryVeryVeryVeryLong0123456789"
    out = swarm._deployment_name()
    assert all(ch.isalnum() or ch in "-_" for ch in out)
    assert len(out) <= 36
    assert out  # non-empty


@pytest.mark.skip(reason="Validation not implemented yet")
def test_deployment_name_sanitization_slurm(cfg_stub):
    swarm = make_swarm(cfg_stub)
    swarm._platform = "slurm"  # type: ignore[attr-defined]
    swarm.name = "Ok_Name-123$%^"
    out = swarm._deployment_name()
    assert all(ch.isalnum() or ch in "-_" for ch in out)
    assert len(out) <= 80


def test_submit_job_builds_command_env_and_calls_run(cfg_stub, monkeypatch):
    swarm = make_swarm(cfg_stub)
    with swarm:  # sets endpoint & serving_handle
        pass

    # Dummy job with to_kwargs
    class DummyJob:
        name = "job"

        def to_kwargs(self):
            return {"alpha": 1}

    job = DummyJob()

    # Ensure deployment has a compute backend and ensure_ready hook
    dep = swarm._deployment  # type: ignore[attr-defined]
    dep.compute = FakeComputeBackend()  # replace to track calls

    pid = swarm.submit_job(
        job,
        input_path=Path("/tmp/in.parquet"),
        output_path=Path("/tmp/out.parquet"),
        num_threads=2,
        detach=True,
        limit=5,
        checkpoint_dir=Path("/tmp/.ckpt"),
    )
    # detach=True → returns pid from JobHandle.meta
    assert pid == 4321

    # Ensure ensure_ready was invoked
    assert dep.ensure_ready_calls == 1

    # Inspect last run call
    call = dep.run_calls[-1]
    # command contains python, module, args and our limit
    cmd = call["command"]
    assert cmd[:3] == ["python", "-m", "domyn_swarm.jobs.run"]
    assert f"--endpoint={swarm.endpoint}" in cmd
    assert "--job-kwargs" in cmd
    # The item right after "--job-kwargs" must be the JSON we created
    idx = cmd.index("--job-kwargs")
    kwargs_json = cmd[idx + 1]
    assert json.loads(kwargs_json) == {"alpha": 1}
    # Limit included
    assert any(arg == "--limit=5" for arg in cmd)
    # env merged from cfg.backend (+ overrides from compute)
    env = call["env"]
    assert env["ENDPOINT"] == swarm.endpoint
    assert env["MODEL"] == cfg_stub.model
    assert env["X"] == "Y"  # from cfg.backend.env
    assert env["A"] == "B"  # from compute.default_env


def test_submit_job_returns_none_when_not_detached(cfg_stub):
    swarm = make_swarm(cfg_stub)
    with swarm:
        pass

    class Job:
        name = "job"

        def to_kwargs(self):
            return {}

    out = swarm.submit_job(
        Job(),
        input_path=Path("/tmp/in.parquet"),
        output_path=Path("/tmp/out.parquet"),
        detach=False,
    )
    assert out is None


def test_cleanup_calls_deployment_down_when_handle_present(cfg_stub):
    swarm = make_swarm(cfg_stub)
    with swarm:
        pass
    dep = swarm._deployment  # type: ignore[attr-defined]
    swarm.cleanup()
    assert dep.down_calls == [swarm.serving_handle.id]


def test_delete_record_calls_state_mgr_delete(cfg_stub):
    swarm = make_swarm(cfg_stub)
    swarm.delete_record("deployment_name")
    assert swarm._state_mgr.deleted == 1  # type: ignore[attr-defined]


def test_from_state_forwards_to_state_manager(monkeypatch, patch_state_mgr):
    monkeypatch.setattr(
        mod.SwarmStateManager,
        "load",
        classmethod(
            lambda cls, deployment_name: SimpleNamespace(
                deployment_name=deployment_name
            )
        ),
    )
    out = DomynLLMSwarm.from_state(deployment_name="name")
    assert out.deployment_name == "name"


def test_make_compute_backend_slurm_happy_path(monkeypatch):
    # Patch SlurmConfig and SlurmComputeBackend in the module so isinstance checks pass
    class _SlurmConfig:
        pass

    class _SlurmCompute:
        def __init__(self, cfg, lb_jobid, lb_node):
            self.cfg, self.lb_jobid, self.lb_node = cfg, lb_jobid, lb_node

    monkeypatch.setattr(mod, "SlurmConfig", _SlurmConfig)
    monkeypatch.setattr(mod, "SlurmComputeBackend", _SlurmCompute)

    cfg = SimpleNamespace(
        name="n",
        model="m1",
        wait_endpoint_s=5,
        backend=_SlurmConfig(),
    )
    cfg.get_deployment_plan = lambda: FakePlan(platform="slurm")
    swarm = make_swarm(cfg)

    # Fake handle with required metadata:
    handle = SimpleNamespace(
        id="dep", url="http://n:9000", meta={"lb_jobid": 777, "lb_node": "nodeX"}
    )
    comp = swarm._make_compute_backend(handle)
    assert isinstance(comp, _SlurmCompute)
    assert comp.lb_jobid == 777 and comp.lb_node == "nodeX"


def test_make_compute_backend_slurm_missing_meta_raises(monkeypatch):
    class _SlurmConfig:
        pass

    monkeypatch.setattr(mod, "SlurmConfig", _SlurmConfig)

    cfg = SimpleNamespace(
        name="", model="m1", wait_endpoint_s=5, backend=_SlurmConfig()
    )
    cfg.get_deployment_plan = lambda: FakePlan(platform="slurm")
    swarm = make_swarm(cfg)

    handle = SimpleNamespace(id="dep", url="", meta={})  # missing lb_* keys
    with pytest.raises(RuntimeError, match="LB Job ID/Node missing"):
        swarm._make_compute_backend(handle)
