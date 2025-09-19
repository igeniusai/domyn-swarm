import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

# SUT module (adjust path if your file lives elsewhere)
import domyn_swarm.backends.compute.slurm as mod
from domyn_swarm.backends.compute.slurm import SlurmComputeBackend
from domyn_swarm.platform.protocols import JobStatus


# ----------------------------
# Fake SrunCommandBuilder
# ----------------------------
class FakeBuilder:
    last = None

    def __init__(self, cfg, jobid, nodelist):
        self.cfg = cfg
        self.jobid = jobid
        self.nodelist = nodelist
        self.env = {}
        self._exe = None
        FakeBuilder.last = self

    def with_env(self, env):
        self.env.update(env)
        return self

    def build(self, exe):
        # Return a concrete srun + flags + the exe tail
        self._exe = list(exe)
        return [
            "srun",
            f"--jobid={self.jobid}",
            f"--nodelist={self.nodelist}",
            *self._exe,
        ]


# ----------------------------
# Fixtures
# ----------------------------
@pytest.fixture(autouse=True)
def patch_builder(monkeypatch):
    # Replace the builder used inside the module
    monkeypatch.setattr(mod, "SrunCommandBuilder", FakeBuilder)
    # Silence rich printing
    monkeypatch.setattr(mod, "rprint", lambda *a, **k: None)
    yield


def _mk_cfg(venv_path=None):
    return SimpleNamespace(venv_path=venv_path)


# ----------------------------
# submit(detach=True)
# ----------------------------
def test_submit_detach_uses_popen_and_returns_running(monkeypatch):
    popen_calls = {}

    class FakePopen:
        def __init__(self, cmd, stdout, stderr, start_new_session, close_fds):
            popen_calls["cmd"] = cmd
            popen_calls["stdout"] = stdout
            popen_calls["stderr"] = stderr
            popen_calls["start_new_session"] = start_new_session
            popen_calls["close_fds"] = close_fds
            self.pid = 5555

    monkeypatch.setattr(mod.subprocess, "Popen", FakePopen)

    be = SlurmComputeBackend(cfg=_mk_cfg(), lb_jobid=123, lb_node="nodeA")

    handle = be.submit(
        name="my-job",
        image=None,
        command=["python", "-c", "print(1)"],
        env={"A": "B"},
        detach=True,
    )

    # Builder captured env & exe
    b = FakeBuilder.last
    assert b is not None
    assert b.env == {"A": "B"}
    assert b._exe == ["python", "-c", "print(1)"]

    # Popen got the cmd returned by build
    assert popen_calls["cmd"] == [
        "srun",
        "--jobid=123",
        "--nodelist=nodeA",
        "python",
        "-c",
        "print(1)",
    ]
    # sanity on flags passed
    assert popen_calls["start_new_session"] is True
    assert popen_calls["close_fds"] is True

    # Handle fields
    assert handle.status is JobStatus.RUNNING
    assert handle.meta["pid"] == 5555
    assert handle.id == "5555"
    assert handle.meta["cmd"] == shlex.join(popen_calls["cmd"])  # joined string


# ----------------------------
# submit(detach=False)
# ----------------------------
def test_submit_sync_uses_run_and_returns_succeeded(monkeypatch):
    run_calls = {}

    def fake_run(cmd, check):
        run_calls["cmd"] = cmd
        run_calls["check"] = check
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    be = SlurmComputeBackend(cfg=_mk_cfg(), lb_jobid=42, lb_node="n1")

    handle = be.submit(
        name="job-1",
        image=None,
        command=["echo", "ok"],
        env=None,
        detach=False,
    )

    # subprocess.run invoked with builder's cmd and check=True
    assert run_calls["cmd"] == ["srun", "--jobid=42", "--nodelist=n1", "echo", "ok"]
    assert run_calls["check"] is True

    # Handle
    assert handle.status is JobStatus.SUCCEEDED
    assert handle.id == "job-1"
    assert handle.meta["cmd"] == shlex.join(run_calls["cmd"])


# ----------------------------
# default_python
# ----------------------------
def test_default_python_uses_venv_when_dir_exists(tmp_path: Path):
    venv_dir = tmp_path / "venv"
    venv_dir.mkdir()
    # It only checks is_dir(); we don't need to create bin/python file
    cfg = _mk_cfg(venv_path=venv_dir)

    be = SlurmComputeBackend(cfg=cfg, lb_jobid=1, lb_node="n")
    out = be.default_python(cfg)
    assert out == str(venv_dir / "bin" / "python")


def test_default_python_falls_back_to_mixin(monkeypatch):
    # Patch the mixin default to a sentinel to ensure super() is called
    monkeypatch.setattr(
        mod.DefaultComputeMixin,
        "default_python",
        lambda self, cfg: "SENTINEL_PY",
        raising=False,
    )

    cfg = _mk_cfg(venv_path=None)
    be = SlurmComputeBackend(cfg=cfg, lb_jobid=1, lb_node="n")
    assert be.default_python(cfg) == "SENTINEL_PY"


# ----------------------------
# wait()
# ----------------------------
def test_wait_returns_status_if_no_pid():
    be = SlurmComputeBackend(cfg=_mk_cfg(), lb_jobid=1, lb_node="n")
    handle = SimpleNamespace(status=JobStatus.PENDING, meta={})
    assert be.wait(handle) is JobStatus.PENDING


def test_wait_with_pid_returns_succeeded():
    be = SlurmComputeBackend(cfg=_mk_cfg(), lb_jobid=1, lb_node="n")
    handle = SimpleNamespace(status=JobStatus.RUNNING, meta={"pid": 123})
    assert be.wait(handle) is JobStatus.SUCCEEDED


# ----------------------------
# cancel()
# ----------------------------
def test_cancel_sends_term_to_pid(monkeypatch):
    calls = []

    def fake_run(argv, check=False):
        calls.append((tuple(argv), check))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    be = SlurmComputeBackend(cfg=_mk_cfg(), lb_jobid=1, lb_node="n")
    be.cancel(SimpleNamespace(meta={"pid": 777}))
    assert calls == [(("kill", "-TERM", "777"), False)]


def test_cancel_swallows_errors(monkeypatch):
    def fake_run(argv, check=False):
        raise RuntimeError("boom")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    be = SlurmComputeBackend(cfg=_mk_cfg(), lb_jobid=1, lb_node="n")
    # Should not raise
    be.cancel(SimpleNamespace(meta={"pid": 888}))
