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
from dataclasses import dataclass
import shlex
import subprocess
import time
from typing import TYPE_CHECKING

from rich import print as rprint
from rich.syntax import Syntax

from domyn_swarm.backends.compute.slurm_helpers import (
    _build_step_log_paths,
    _build_step_name,
    _cancel_slurm,
    _create_step_id_fifo,
    _probe_slurm,
    _resolve_external_id,
    _terminate_process_group,
    _wait_for_slurm,
    _wait_for_slurm_with_log_stream,
    _wrap_with_step_echo,
)
from domyn_swarm.backends.serving.srun_builder import SrunCommandBuilder
from domyn_swarm.config.swarm import DomynLLMSwarmConfig
from domyn_swarm.helpers.logger import setup_logger
from domyn_swarm.platform.protocols import DefaultComputeMixin, JobHandle, JobProbe, JobStatus

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
    - For `detach=True`, we return a JobHandle with local launcher PID in `meta["pid"]`
      (diagnostic only) and required Slurm step id in `meta["external_id"]`.
    - Reconnect semantics (`wait/cancel/probe`) are external-id only and therefore stable
      across CLI invocations and process boundaries.
    - If step-id resolution fails at submit time, the detached process is terminated and submit
      fails to avoid orphaned non-reconnectable jobs.
    """

    cfg: "SlurmConfig"
    lb_jobid: int
    lb_node: str

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
        step_log_paths = _build_step_log_paths(extras, step_name) if detach else None
        if step_name:
            builder = builder.with_extra_args([f"--job-name={step_name}"])
        if step_log_paths:
            builder = builder.with_extra_args(
                [
                    f"--output={step_log_paths['stdout']}",
                    f"--error={step_log_paths['stderr']}",
                ]
            )
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
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                start_new_session=True,
                close_fds=True,
            )
            if proc.pid is None:
                raise RuntimeError("Detached process did not get a PID.")
            external_id = _resolve_external_id(
                stdout=None,
                job_id=self.lb_jobid,
                step_name=step_name,
                step_id_fifo=step_id_fifo,
            )
            if not external_id:
                with contextlib.suppress(Exception):
                    _terminate_process_group(proc.pid, grace_s=3.0)
                raise RuntimeError(
                    "Failed to resolve detached Slurm step id (external_id); "
                    "job was terminated to preserve reconnect semantics."
                )
            return JobHandle(
                id=external_id,
                status=JobStatus.RUNNING,
                meta={
                    "pid": proc.pid,
                    "external_id": external_id,
                    "log_paths": step_log_paths or _extract_log_paths(cmd),
                    "step_id_fifo": str(step_id_fifo) if step_id_fifo else None,
                    "cmd": shlex.join(cmd),
                },
            )

        # synchronous
        subprocess.run(cmd, check=True)
        return JobHandle(id=name, status=JobStatus.SUCCEEDED, meta={"cmd": shlex.join(cmd)})

    def probe(self, handle: JobHandle) -> JobProbe:
        """Probe detached job status without blocking.

        Args:
            handle: Job handle to probe.

        Returns:
            Best-effort status probe payload.
        """
        external_id = handle.meta.get("external_id")
        if external_id:
            probe = _probe_slurm(str(external_id))
            handle.status = probe.status
            return probe

        if handle.status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}:
            return JobProbe(status=handle.status, source="local")

        handle.status = JobStatus.FAILED
        return JobProbe(
            status=JobStatus.FAILED,
            raw_status="MISSING_EXTERNAL_ID",
            source="local",
            error=(
                "Cannot probe job without external_id. "
                "Use a reconnectable handle containing Slurm step id."
            ),
        )

    def wait(
        self, handle: JobHandle, *, stream_logs: bool = True, timeout: float | None = None
    ) -> JobStatus:
        """Wait for a detached job to complete.

        Args:
            handle: Job handle returned by `submit(detach=True)`.
            stream_logs: If True, stream detached log files when `log_paths` metadata is available.

        Returns:
            Normalized job status.
        """
        external_id = handle.meta.get("external_id")
        if external_id:
            endpoint_cfg = getattr(self.cfg, "endpoint", None)
            poll_s = float(getattr(endpoint_cfg, "poll_interval", 10))
            log_paths = handle.meta.get("log_paths")
            if stream_logs and isinstance(log_paths, dict):
                status = _wait_for_slurm_with_log_stream(
                    external_id,
                    stdout_path=_coerce_log_path(log_paths.get("stdout")),
                    stderr_path=_coerce_log_path(log_paths.get("stderr")),
                    timeout=timeout,
                    poll_s=poll_s,
                )
            else:
                status = _wait_for_slurm(
                    external_id,
                    timeout=timeout,
                    poll_s=poll_s,
                )
            handle.status = status
            return status

        if stream_logs:
            logger.debug("Slurm detached wait ignores stream_logs and uses scheduler polling.")

        if handle.status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}:
            return handle.status

        logger.warning(
            "Cannot wait() without external_id for non-terminal job; "
            "use a reconnectable handle containing Slurm step id."
        )
        handle.status = JobStatus.FAILED
        return handle.status

    def cancel(self, handle: JobHandle) -> None:
        """Cancel a detached job via its Slurm step id.

        Reconnect semantics are external-id only: a detached handle always carries the
        resolved Slurm step id in ``meta["external_id"]`` (submit fails otherwise), so
        cancellation is stable across CLI invocations and process boundaries.

        Args:
            handle: Job handle returned by ``submit(detach=True)``.
        """
        external_id = handle.meta.get("external_id")
        if not external_id:
            logger.warning(
                "Cannot cancel job without external_id; "
                "use a reconnectable handle containing the Slurm step id."
            )
            return

        with contextlib.suppress(Exception):
            _cancel_slurm(external_id)
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


def _extract_log_paths(cmd: Sequence[str]) -> dict[str, str]:
    """Extract Slurm log paths from rendered ``srun`` arguments.

    Args:
        cmd: Final command list passed to ``subprocess``.

    Returns:
        Dictionary containing optional ``stdout`` and ``stderr`` file paths.
    """
    log_paths: dict[str, str] = {}
    for arg in cmd:
        if arg.startswith("--output="):
            log_paths["stdout"] = arg.split("=", 1)[1]
        elif arg.startswith("--error="):
            log_paths["stderr"] = arg.split("=", 1)[1]
    return log_paths


def _coerce_log_path(value: object) -> str | None:
    """Normalize optional log path payload.

    Args:
        value: Raw value from handle metadata.

    Returns:
        Log path string when valid, else ``None``.
    """
    if value is None:
        return None
    path = str(value).strip()
    return path or None
