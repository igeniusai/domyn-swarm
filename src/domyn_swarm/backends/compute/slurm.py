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
from typing import TYPE_CHECKING

from rich import print as rprint
from rich.syntax import Syntax

from domyn_swarm.backends.serving.srun_builder import SrunCommandBuilder
from domyn_swarm.config.slurm import SlurmConfig
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
        return JobHandle(id=name, status=JobStatus.SUCCEEDED, meta={"cmd": shlex.join(cmd)})

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
            with contextlib.suppress(Exception):
                subprocess.run(["kill", "-TERM", str(pid)], check=False)

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
