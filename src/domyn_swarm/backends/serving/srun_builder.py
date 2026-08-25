# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
import os

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
        # If we're already inside a Slurm allocation (i.e. SLURM_JOB_ID is set),
        # avoid pinning execution to the load-balancer allocation/node. This prevents
        # large data jobs from running on the LB node when launched from a Slurm job.
        in_slurm_allocation = (os.getenv("SLURM_JOB_ID") or os.getenv("SLURM_JOBID")) is not None
        require_allocated = getattr(self.cfg.endpoint, "require_allocated_node", False)
        if require_allocated and not in_slurm_allocation:
            raise ValueError(
                "srun requires running inside a Slurm allocation when "
                "`require_allocated_node` is enabled."
            )

        cmd = [
            "srun",
            f"--ntasks={ntasks}",
            "--nodes=1",
            "--overlap",
        ]
        if not in_slurm_allocation:
            cmd.insert(1, f"--jobid={self.jobid}")
            cmd.insert(2, f"--nodelist={self.nodelist}")

        if (
            any("--mem" in arg for arg in self.extra_args) is False
            and self.cfg.endpoint.mem is not None
        ):
            cmd.append(f"--mem={self.cfg.endpoint.mem}")

        if (
            any("--cpus-per-task" in arg for arg in self.extra_args) is False
            and self.cfg.endpoint.cpus_per_task is not None
        ):
            cmd.append(f"--cpus-per-task={self.cfg.endpoint.cpus_per_task}")

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
