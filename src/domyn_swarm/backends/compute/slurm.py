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

from collections.abc import Mapping, Sequence
import contextlib
from dataclasses import dataclass, field
import shlex
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING

from rich import print as rprint
from rich.syntax import Syntax

from domyn_swarm.backends.compute.slurm_helpers import (
    _build_step_name,
    _cancel_slurm,
    _create_step_id_fifo,
    _normalize_returncode,
    _pid_exists,
    _resolve_external_id,
    _stream_text_lines,
    _terminate_process_group,
    _wait_for_slurm,
    _wrap_with_step_echo,
)
from domyn_swarm.backends.serving.srun_builder import SrunCommandBuilder
from domyn_swarm.config.swarm import DomynLLMSwarmConfig
from domyn_swarm.helpers.logger import setup_logger
from domyn_swarm.platform.protocols import DefaultComputeMixin, JobHandle, JobStatus

if TYPE_CHECKING:
    from domyn_swarm.config.slurm import SlurmConfig

logger = setup_logger(__name__)


@dataclass
class SlurmComputeBackend(DefaultComputeMixin):  # type: ignore[misc]
    """Compute backend using `srun` inside the LB allocation.

    Notes
    -----
    - `image` is unused for Slurm; your venv/python is used directly.
    - `resources` maps to srun flags via your SrunCommandBuilder configuration.
    - For `detach=True`, we return a JobHandle with local `Popen` PID in `meta["pid"]` and
      best-effort resolve the Slurm step id into `meta["external_id"]` using a FIFO channel
      (preferred) or `squeue`/`sacct` fallback.
    - Cancellation uses the Slurm external id when available; otherwise it falls back to
      `os.killpg(pid, ...)` to terminate the whole `srun` process group.
    """

    cfg: "SlurmConfig"
    lb_jobid: int
    lb_node: str
    _procs: dict[int, subprocess.Popen[str]] = field(default_factory=dict, init=False, repr=False)

    def submit(
        self,
        *,
        name: str,
        image: str | None,
        command: Sequence[str],
        env: Mapping[str, str] | None = None,
        resources: dict | None = None,
        detach: bool = False,
        nshards: int | None = None,
        shard_id: int | None = None,
        extras: dict | None = None,
    ) -> JobHandle:
        builder = SrunCommandBuilder(self.cfg, self.lb_jobid, self.lb_node)
        if env:
            builder = builder.with_env(dict(env))

        if resources:
            # Resources should be a dict like {"cpus_per_task": 4, "mem": "16G", ...}
            extra_args: list[str] = []
            for key, value in resources.items():
                flag = f"--{key.replace('_', '-')}"
                if value is True:
                    extra_args.append(flag)
                elif value is False or value is None:
                    continue
                else:
                    extra_args.append(f"{flag}={value}")
            builder = builder.with_extra_args(extra_args)

        step_name = _build_step_name(name) if detach else None
        step_id_fifo = _create_step_id_fifo(extras, step_name) if detach else None
        if step_name:
            builder = builder.with_extra_args([f"--job-name={step_name}"])
        wrapped = _wrap_with_step_echo(command, step_id_fifo) if detach else [*map(str, command)]
        cmd = builder.build(wrapped)

        full_cmd = shlex.join(cmd)

        syntax = Syntax(
            full_cmd,
            "bash",
            line_numbers=False,
            word_wrap=True,
            indent_guides=True,
            padding=1,
        )
        rprint(syntax)

        if detach:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
                close_fds=True,
            )
            if proc.pid is None:
                raise RuntimeError("Detached process did not get a PID.")
            self._procs[proc.pid] = proc
            external_id = _resolve_external_id(
                stdout=proc.stdout,
                job_id=self.lb_jobid,
                step_name=step_name,
                step_id_fifo=step_id_fifo,
            )
            return JobHandle(
                id=external_id or str(proc.pid),
                status=JobStatus.RUNNING,
                meta={
                    "pid": proc.pid,
                    "external_id": external_id,
                    "step_id_fifo": str(step_id_fifo) if step_id_fifo else None,
                    "cmd": shlex.join(cmd),
                },
            )

        # synchronous
        subprocess.run(cmd, check=True)
        return JobHandle(id=name, status=JobStatus.SUCCEEDED, meta={"cmd": shlex.join(cmd)})

    def wait(
        self, handle: JobHandle, *, stream_logs: bool = True, timeout: float | None = None
    ) -> JobStatus:
        """Wait for a detached job to complete.

        Args:
            handle: Job handle returned by `submit(detach=True)`.
            stream_logs: If True, stream combined stdout/stderr while waiting.

        Returns:
            Normalized job status.
        """
        external_id = handle.meta.get("external_id")
        if external_id:
            endpoint_cfg = getattr(self.cfg, "endpoint", None)
            status = _wait_for_slurm(
                external_id,
                timeout=timeout,
                poll_s=float(getattr(endpoint_cfg, "poll_interval", 10)),
            )
            handle.status = status
            return status

        pid_raw = handle.meta.get("pid")
        if pid_raw is None:
            return handle.status

        try:
            pid = int(pid_raw)
        except (TypeError, ValueError):
            logger.warning("Invalid pid in job handle meta: %r", pid_raw)
            handle.status = JobStatus.FAILED
            return handle.status

        proc = self._procs.get(pid)
        if proc is None:
            # If we don't have the Popen object, we can't observe the exit code. Provide
            # best-effort "still running?" detection and otherwise mark as FAILED.
            if _pid_exists(pid):
                return JobStatus.RUNNING
            logger.warning(
                "Cannot wait() for pid=%s: process is gone and exit status is unknown "
                "(likely created in a different process).",
                pid,
            )
            handle.status = JobStatus.FAILED
            return handle.status

        streamer: threading.Thread | None = None
        if stream_logs and proc.stdout is not None:
            streamer = threading.Thread(
                target=_stream_text_lines,
                args=(proc.stdout, sys.stdout),
                daemon=True,
            )
            streamer.start()

        rc = proc.wait(timeout=timeout)
        if streamer is not None:
            streamer.join(timeout=2.0)

        self._procs.pop(pid, None)

        handle.meta["returncode"] = rc
        handle.status = _normalize_returncode(rc)
        return handle.status

    def cancel(self, handle: JobHandle) -> None:
        """Cancel a detached job.

        This sends SIGTERM to the detached job's process group (covers `srun` and its children)
        and escalates to SIGKILL after a short grace period.
        """
        external_id = handle.meta.get("external_id")
        if external_id:
            with contextlib.suppress(Exception):
                _cancel_slurm(external_id)
            handle.status = JobStatus.CANCELLED
            handle.meta["cancelled_at"] = time.time()
            return

        pid_raw = handle.meta.get("pid")
        if pid_raw is None:
            return

        try:
            pid = int(pid_raw)
        except (TypeError, ValueError):
            logger.warning("Invalid pid in job handle meta: pid=%r", pid_raw)
            return

        with contextlib.suppress(Exception):
            _terminate_process_group(pid, grace_s=10.0)

        handle.status = JobStatus.CANCELLED
        handle.meta["cancelled_at"] = time.time()

    def default_python(self, cfg: "DomynLLMSwarmConfig") -> str:
        if self.cfg.venv_path and self.cfg.venv_path.is_dir():
            return str(self.cfg.venv_path / "bin" / "python")
        return super().default_python(cfg)

    def default_resources(self, cfg):
        if self.cfg.endpoint.cpus_per_task is not None or self.cfg.endpoint.mem is not None:
            res: dict = {}
            if self.cfg.endpoint.cpus_per_task is not None:
                res["cpus_per_task"] = self.cfg.endpoint.cpus_per_task
            if self.cfg.endpoint.mem is not None:
                res["mem"] = self.cfg.endpoint.mem
            return res
        return super().default_resources(cfg)
