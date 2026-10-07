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
        """Build the srun command with the configured parameters.

        Inside a Slurm allocation the step runs in that allocation; outside one
        it is pinned to the load-balancer job and node.

        Raises:
            ValueError: If `require_allocated_node` is enabled and the caller is
                outside a Slurm allocation or inside the load-balancer one.
        """
        current_job = os.getenv("SLURM_JOB_ID") or os.getenv("SLURM_JOBID")
        in_slurm_allocation = current_job is not None
        require_allocated = getattr(self.cfg.endpoint, "require_allocated_node", False)
        if require_allocated and not in_slurm_allocation:
            raise ValueError(
                "srun requires running inside a Slurm allocation when "
                "`require_allocated_node` is enabled."
            )
        if require_allocated and current_job == str(self.jobid):
            raise ValueError(
                f"srun is running inside the load-balancer allocation (job {self.jobid}). "
                "Submit from a separate Slurm allocation when "
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
