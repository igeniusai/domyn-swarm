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

from __future__ import annotations

from collections.abc import Sequence

from domyn_swarm.config.slurm import SlurmConfig


class SrunCommandBuilder:
    """Builder for constructing srun commands with various configurations."""

    def __init__(self, cfg: SlurmConfig, jobid: int, nodelist: str):
        self.cfg = cfg
        self.jobid = jobid
        self.nodelist = nodelist
        self.env: dict[str, str] = {}
        self.mail_user: str | None = None
        self.extra_args: list[str] = []

    def with_env(self, env: dict[str, str]) -> SrunCommandBuilder:
        self.env.update(env)
        return self

    def with_mail(self, user: str) -> SrunCommandBuilder:
        self.mail_user = user
        return self

    def with_extra_args(self, args: list[str]) -> SrunCommandBuilder:
        self.extra_args.extend(args)
        return self

    def build(self, exe: Sequence[str], ntasks: int = 1) -> list[str]:
        """
        Build the srun command with the configured parameters.

        :param exe: The executable command to run.
        :param ntasks: Number of tasks to run.
        :return: A list representing the srun command.
        """
        cmd = [
            "srun",
            f"--jobid={self.jobid}",
            f"--nodelist={self.nodelist}",
            f"--ntasks={ntasks}",
            "--overlap",
            f"--mem={self.cfg.endpoint.mem}",
            f"--cpus-per-task={self.cfg.endpoint.cpus_per_task}",
        ]

        if self.env:
            export_env = ",".join(f"{k}={v}" for k, v in self.env.items())
            cmd.append(f"--export=ALL,{export_env}")
        else:
            cmd.append("--export=ALL")

        if self.mail_user:
            cmd.append(f"--mail-user={self.mail_user}")
            cmd.append("--mail-type=END,FAIL")

        if self.extra_args:
            cmd.extend(self.extra_args)

        cmd.extend(exe)
        return cmd
