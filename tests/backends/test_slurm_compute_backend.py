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

from pathlib import Path
import shlex
import sys
import time
from types import SimpleNamespace

import pytest

# SUT module (adjust path if your file lives elsewhere)
import domyn_swarm.backends.compute.slurm as mod
from domyn_swarm.backends.compute.slurm import SlurmComputeBackend
import domyn_swarm.backends.compute.slurm_helpers as helpers
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
        self.extra_args = []
        self._exe = None
        FakeBuilder.last = self

    def with_env(self, env):
        self.env.update(env)
        return self

    def with_extra_args(self, args):
        self.extra_args.extend(args)
        return self

    def build(self, exe):
        # Return a concrete srun + flags + the exe tail
        self._exe = list(exe)
        return [
            "srun",
            f"--jobid={self.jobid}",
            f"--nodelist={self.nodelist}",
            *self.extra_args,
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


def _mk_swarm_cfg():
    return SimpleNamespace(backend=_mk_cfg())


# ----------------------------
# FIFO helpers
# ----------------------------
def test_create_step_id_fifo_uses_swarm_jobs_dir(tmp_path, monkeypatch):
    created = {}

    def fake_mkfifo(path, mode):
        created["path"] = Path(path)
        created["mode"] = mode
        Path(path).write_text("")

    monkeypatch.setattr(helpers.os, "mkfifo", fake_mkfifo)

    swarm_dir = tmp_path / "swarms" / "demo"
    extras = {"swarm_directory": str(swarm_dir)}
    step_name = "my-step"

    fifo = mod._create_step_id_fifo(extras, step_name)

    assert fifo == swarm_dir / "jobs" / f"{step_name}.fifo"
    assert created["path"] == fifo
    assert created["mode"] == 0o600
    assert fifo.exists()


# ----------------------------
# submit(detach=True)
# ----------------------------
def test_submit_detach_uses_popen_and_returns_running(monkeypatch):
    popen_calls = {}
    wait_calls = {}

    class FakePopen:
        def __init__(self, cmd, stdin, stdout, stderr, text, start_new_session, close_fds):
            popen_calls["cmd"] = cmd
            popen_calls["stdin"] = stdin
            popen_calls["stdout"] = stdout
            popen_calls["stderr"] = stderr
            popen_calls["text"] = text
            popen_calls["start_new_session"] = start_new_session
            popen_calls["close_fds"] = close_fds
            self.pid = 5555
            self.stdout = None

    monkeypatch.setattr(mod.subprocess, "Popen", FakePopen)

    def fake_wait_for_step_id(**kwargs):
        wait_calls["kwargs"] = kwargs
        return "123.0"

    monkeypatch.setattr(helpers, "_wait_for_step_id", fake_wait_for_step_id)

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
    assert b._exe[0:2] == ["bash", "-lc"]
    assert 'echo "${SLURM_JOB_ID}.${SLURM_STEP_ID}"' in b._exe[2]

    # Popen got the cmd returned by build (wrapped with bash -lc)
    assert popen_calls["cmd"][:4] == [
        "srun",
        "--jobid=123",
        "--nodelist=nodeA",
        b.extra_args[0],
    ]
    assert popen_calls["cmd"][4] == "bash"
    assert popen_calls["cmd"][5] == "-lc"
    assert 'echo "${SLURM_JOB_ID}.${SLURM_STEP_ID}"' in popen_calls["cmd"][6]
    assert "exec python -c 'print(1)'" in popen_calls["cmd"][6]
    # sanity on flags passed
    assert popen_calls["start_new_session"] is True
    assert popen_calls["close_fds"] is True
    assert popen_calls["text"] is True

    # Handle fields
    assert handle.status is JobStatus.RUNNING
    assert handle.meta["pid"] == 5555
    assert handle.meta["external_id"] == "123.0"
    assert handle.id == "123.0"
    assert handle.meta["cmd"] == shlex.join(popen_calls["cmd"])  # joined string
    assert wait_calls["kwargs"]["job_id"] == 123
    assert wait_calls["kwargs"]["step_name"] == b.extra_args[0].split("=", 1)[1]


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


def test_submit_adds_resource_flags(monkeypatch):
    run_calls = {}

    def fake_run(cmd, check):
        run_calls["cmd"] = cmd
        run_calls["check"] = check
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    be = SlurmComputeBackend(cfg=_mk_cfg(), lb_jobid=11, lb_node="n-res")
    handle = be.submit(
        name="job-res",
        image=None,
        command=["echo", "hi"],
        resources={"cpus_per_task": 8, "mem": "24G", "exclusive": True, "tmp": False},
        detach=False,
    )

    # Extra args passed to builder in order and False skipped
    b = FakeBuilder.last
    assert b.extra_args == ["--cpus-per-task=8", "--mem=24G", "--exclusive"]

    # subprocess.run invoked with extra args before the exe tail
    assert run_calls["cmd"] == [
        "srun",
        "--jobid=11",
        "--nodelist=n-res",
        "--cpus-per-task=8",
        "--mem=24G",
        "--exclusive",
        "echo",
        "hi",
    ]
    assert handle.status is JobStatus.SUCCEEDED
    assert handle.id == "job-res"


# ----------------------------
# default_python
# ----------------------------
def test_default_python_uses_venv_when_dir_exists(tmp_path: Path):
    venv_dir = tmp_path / "venv"
    venv_dir.mkdir()
    # It only checks is_dir(); we don't need to create bin/python file
    cfg = _mk_cfg(venv_path=venv_dir)

    be = SlurmComputeBackend(cfg=cfg, lb_jobid=1, lb_node="n")
    out = be.default_python(_mk_swarm_cfg())
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


def test_wait_with_external_id_uses_slurm_poll(monkeypatch):
    be = SlurmComputeBackend(cfg=_mk_cfg(), lb_jobid=1, lb_node="n")
    handle = SimpleNamespace(status=JobStatus.RUNNING, meta={"external_id": "123.0"})
    monkeypatch.setattr(mod, "_wait_for_slurm", lambda *a, **k: JobStatus.SUCCEEDED)
    assert be.wait(handle) is JobStatus.SUCCEEDED


def test_wait_with_pid_but_no_popen_returns_running_if_pid_exists(monkeypatch):
    be = SlurmComputeBackend(cfg=_mk_cfg(), lb_jobid=1, lb_node="n")
    handle = SimpleNamespace(status=JobStatus.RUNNING, meta={"pid": 123})
    monkeypatch.setattr(mod, "_pid_exists", lambda pid: True)
    assert be.wait(handle) is JobStatus.RUNNING


def test_wait_with_pid_but_no_popen_returns_failed_if_pid_missing(monkeypatch):
    be = SlurmComputeBackend(cfg=_mk_cfg(), lb_jobid=1, lb_node="n")
    handle = SimpleNamespace(status=JobStatus.RUNNING, meta={"pid": 123})
    monkeypatch.setattr(mod, "_pid_exists", lambda pid: False)
    assert be.wait(handle) is JobStatus.FAILED


def test_wait_tracks_detached_popen_and_returns_succeeded():
    cfg = _mk_cfg()
    be = SlurmComputeBackend(cfg=cfg, lb_jobid=1, lb_node="n")

    proc = mod.subprocess.Popen(
        [sys.executable, "-c", "print('ok')"],
        stdout=mod.subprocess.PIPE,
        stderr=mod.subprocess.STDOUT,
        text=True,
        start_new_session=True,
        close_fds=True,
    )
    assert proc.pid is not None
    be._procs[proc.pid] = proc

    handle = SimpleNamespace(status=JobStatus.RUNNING, meta={"pid": proc.pid})
    status = be.wait(handle, stream_logs=False)
    assert status is JobStatus.SUCCEEDED
    assert handle.meta["returncode"] == 0
    assert proc.pid not in be._procs


# ----------------------------
# cancel()
# ----------------------------
def test_cancel_terminates_process_group(monkeypatch):
    calls = []
    monkeypatch.setattr(
        mod,
        "_terminate_process_group",
        lambda pgid, grace_s=10.0: calls.append(pgid),
    )
    be = SlurmComputeBackend(cfg=_mk_cfg(), lb_jobid=1, lb_node="n")
    handle = SimpleNamespace(status=JobStatus.RUNNING, meta={"pid": 777})
    be.cancel(handle)
    assert calls == [777]
    assert handle.status is JobStatus.CANCELLED


def test_cancel_external_id_calls_scancel(monkeypatch):
    calls = []
    monkeypatch.setattr(mod, "_cancel_slurm", lambda ext: calls.append(ext))
    be = SlurmComputeBackend(cfg=_mk_cfg(), lb_jobid=1, lb_node="n")
    handle = SimpleNamespace(status=JobStatus.RUNNING, meta={"external_id": "123.0"})
    be.cancel(handle)
    assert calls == ["123.0"]
    assert handle.status is JobStatus.CANCELLED


def test_cancel_swallows_errors(monkeypatch):
    def fake_killpg(pgid, grace_s=10.0):
        raise RuntimeError("boom")

    monkeypatch.setattr(mod, "_terminate_process_group", fake_killpg)

    be = SlurmComputeBackend(cfg=_mk_cfg(), lb_jobid=1, lb_node="n")
    # Should not raise
    be.cancel(SimpleNamespace(meta={"pid": 888}))


def test_cancel_then_wait_returns_cancelled():
    cfg = _mk_cfg()
    be = SlurmComputeBackend(cfg=cfg, lb_jobid=1, lb_node="n")

    proc = mod.subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=mod.subprocess.PIPE,
        stderr=mod.subprocess.STDOUT,
        text=True,
        start_new_session=True,
        close_fds=True,
    )
    assert proc.pid is not None
    be._procs[proc.pid] = proc

    handle = SimpleNamespace(status=JobStatus.RUNNING, meta={"pid": proc.pid})
    be.cancel(handle)

    # Give the signal a moment to land on slower CI boxes.
    time.sleep(0.1)
    status = be.wait(handle, stream_logs=False)
    assert status is JobStatus.CANCELLED
