from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence
from domyn_swarm.models.swarm import DomynLLMSwarmConfig


class SrunCommandBuilder:
    """Builder for constructing srun commands with various configurations."""
    def __init__(self, cfg: DomynLLMSwarmConfig, jobid: int, nodelist: str):
        self.cfg = cfg
        self.jobid = jobid
        self.nodelist = nodelist
        self.env: dict[str, str] = {}
        self.mail_user: Optional[str] = None
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
            f"--mem={self.cfg.driver.mem}",
            f"--cpus-per-task={self.cfg.driver.cpus_per_task}",
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