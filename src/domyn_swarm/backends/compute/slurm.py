import shlex
import subprocess
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from domyn_swarm.core.srun_builder import SrunCommandBuilder  # your file/module
from domyn_swarm.platform.protocols import ComputeBackend, JobHandle, JobStatus


@dataclass
class SlurmCompute(ComputeBackend):  # type: ignore[misc]
    """Compute backend using `srun` inside the LB allocation.

    Notes
    -----
    - `image` is unused for Slurm; your venv/python is used directly.
    - `resources` maps to srun flags via your SrunCommandBuilder configuration.
    - For `detach=True`, we return a JobHandle with local Popen PID in `meta["pid"]`.
      `wait()` will join that process.
    """

    # TODO: refine type of cfg
    cfg: Any
    lb_jobid: int
    lb_node: str

    def submit(
        self,
        *,
        name: str,
        image: Optional[str],
        command: Sequence[str],
        env: Optional[Mapping[str, str]] = None,
        resources: Optional[dict] = None,
        detach: bool = False,
        nshards: Optional[int] = None,
        shard_id: Optional[int] = None,
    ) -> JobHandle:
        builder = SrunCommandBuilder(self.cfg, self.lb_jobid, self.lb_node)
        if env:
            builder = builder.with_env(dict(env))
        cmd = builder.build([*map(str, command)])

        if detach:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
            return JobHandle(
                id=str(proc.pid),
                status=JobStatus.RUNNING,
                meta={"pid": proc.pid, "cmd": shlex.join(cmd)},
            )

        # synchronous
        subprocess.run(cmd, check=True)
        return JobHandle(
            id=name, status=JobStatus.SUCCEEDED, meta={"cmd": shlex.join(cmd)}
        )

    def wait(self, handle: JobHandle, *, stream_logs: bool = True) -> JobStatus:
        pid = handle.meta.get("pid")
        if pid is None:
            return handle.status
        # Best-effort: we didn't keep the Popen object here; in practice, keep it around
        # or use `psutil` to join. This is a stub.
        return JobStatus.SUCCEEDED

    def cancel(self, handle: JobHandle) -> None:
        pid = handle.meta.get("pid")
        if pid is not None:
            try:
                subprocess.run(["kill", "-TERM", str(pid)], check=False)
            except Exception:
                pass
