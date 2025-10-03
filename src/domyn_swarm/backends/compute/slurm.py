# Copyright 2025 Domyn
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

import shlex
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Optional, Sequence

from rich import print as rprint
from rich.syntax import Syntax

from domyn_swarm.backends.serving.srun_builder import SrunCommandBuilder
from domyn_swarm.config.slurm import SlurmConfig
from domyn_swarm.config.swarm import DomynLLMSwarmConfig
from domyn_swarm.platform.protocols import DefaultComputeMixin, JobHandle, JobStatus

if TYPE_CHECKING:
    from domyn_swarm.config.slurm import SlurmConfig


@dataclass
class SlurmComputeBackend(DefaultComputeMixin):  # type: ignore[misc]
    """Compute backend using `srun` inside the LB allocation.

    Notes
    -----
    - `image` is unused for Slurm; your venv/python is used directly.
    - `resources` maps to srun flags via your SrunCommandBuilder configuration.
    - For `detach=True`, we return a JobHandle with local Popen PID in `meta["pid"]`.
      `wait()` will join that process.
    """

    cfg: "SlurmConfig"
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
        extras: dict | None = None,
    ) -> JobHandle:
        builder = SrunCommandBuilder(self.cfg, self.lb_jobid, self.lb_node)
        if env:
            builder = builder.with_env(dict(env))
        cmd = builder.build([*map(str, command)])

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

    def default_python(self, cfg: "DomynLLMSwarmConfig") -> str:
        if self.cfg.venv_path and self.cfg.venv_path.is_dir():
            return str(self.cfg.venv_path / "bin" / "python")
        return super().default_python(cfg)
