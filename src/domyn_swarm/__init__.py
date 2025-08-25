import importlib
import json
import logging
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, ValidationInfo, computed_field, field_validator
from rich import print as rprint
from rich.syntax import Syntax

from domyn_swarm import utils
from domyn_swarm.helpers.data import (
    generate_swarm_name,
)
from domyn_swarm.helpers.io import to_path
from domyn_swarm.helpers.logger import setup_logger
from domyn_swarm.jobs import SwarmJob
from domyn_swarm.models.swarm import DomynLLMSwarmConfig

from .core.lb_health_checker import LBHealthChecker
from .core.slurm_driver import SlurmDriver
from .core.srun_builder import SrunCommandBuilder
from .core.state import SwarmStateManager

logger = setup_logger(__name__, level=logging.INFO)


class DomynLLMSwarm(BaseModel):
    """
    Context manager that:
      1. renders and submits one Slurm job as a job array (each job is a replica of the same cluster)
      2. waits for it to COMPLETED/FAILED via sacct
      3. cleans up on error or ^C

    Inside the allocation:
      • SLURM_NODEID 0 runs the LB (nginx) + user driver
      • SLURM_NODEID 1…nodes run the vLLM servers
    """

    name: Optional[str] = Field(default_factory=generate_swarm_name)
    cfg: DomynLLMSwarmConfig
    jobid: Optional[int] = None  # Slurm job id, set after job submission
    lb_jobid: Optional[int] = None  # LB job id, set after job submission
    lb_node: Optional[str] = None  # the node where the LB is running
    endpoint: Optional[str] = None  # LB endpoint, set after job submission
    delete_on_exit: Optional[bool] = (
        False  # Delete the resources for this cluster at the end of the job
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str, info: ValidationInfo) -> str:
        if v is None:
            return cls.model_fields[info.field_name].get_default(
                call_default_factory=True
            )
        return v

    @computed_field
    @property
    def model(self) -> str:
        """
        The model name, either from the config or the job submission.
        If not set, defaults to the config's model.
        """
        return self.cfg.model

    @model.setter
    def model(self, value: str):
        """
        Setter for the model name. This allows setting the model after
        the swarm has been created, e.g., when loading from a state file.
        """
        self.cfg.model = value

    def model_post_init(self, __context: Any) -> None:
        self._slurm = SlurmDriver(self.cfg)
        self._state_mgr = SwarmStateManager(self)
        self._lb_checker = LBHealthChecker(self)

    def __enter__(self):
        self._submit_clusters_job()
        self._wait_for_lb_health()
        self._persist()
        return self  # nothing else to expose in Mode B

    def __exit__(self, exc_type, exc, tb):
        if self.delete_on_exit:
            self.cleanup()

    def _submit_replicas(self, job_name: str) -> int:
        return self._slurm.submit_replicas(job_name)

    def _submit_lb(self, job_name: str) -> int:
        assert self.jobid is not None
        return self._slurm.submit_lb(job_name, self.jobid)

    def submit_script(
        self,
        script_path: Path,
        detach: bool = False,
        extra_args: list[str] | None = None,
    ):
        """
        Submit a user script to the swarm allocation.
        The script will be run on the head node (SLURM_NODEID 0).
        """

        if self.lb_jobid is None:
            raise RuntimeError("LB Job ID is null.")

        if self.lb_node is None:
            raise RuntimeError("LB Node is null.")

        if self.endpoint is None:
            raise RuntimeError("Endpoint is null.")

        if self.cfg.venv_path is None:
            raise RuntimeError("Venv path is None.")

        if self.jobid is None:
            raise RuntimeError("No job submitted yet")
        if not script_path.is_file():
            raise FileNotFoundError(f"Script not found: {script_path}")

        logger.info(f"Submitting user script {script_path} to job {self.jobid}")

        builder = SrunCommandBuilder(self.cfg, self.lb_jobid, self.lb_node).with_env(
            {"ENDPOINT": self.endpoint, "MODEL": self.model}
        )

        if self.cfg.mail_user:
            builder = builder.with_mail(self.cfg.mail_user)

        extra = [] if extra_args is None else extra_args
        cmd = builder.build(
            [
                str(self.cfg.venv_path / "bin" / "python"),
                str(script_path),
                *extra,
            ]
        )

        if detach:
            proc = subprocess.Popen(
                cmd,
                stdout=sys.stdout,
                stderr=sys.stderr,
                start_new_session=True,
                close_fds=True,
            )
            logger.info(f"Detached process with PID {proc.pid}")
            return proc.pid
        else:
            subprocess.run(
                cmd,
                check=True,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )

    def _persist(self):
        self._state_mgr.save()

    def _submit_clusters_job(self):
        if self.name is None:
            raise RuntimeError("Name is null.")

        job_name = self.name
        self.jobid = self._submit_replicas(job_name)
        logger.info(
            f"Submitted Slurm job {self.jobid} with {self.cfg.replicas} replicas"
        )
        self.lb_jobid = self._submit_lb(job_name)
        logger.info(f"Submitted LB job for {self.lb_jobid}")
        self._persist()

    def _get_lb_node(self) -> str:
        if self.lb_jobid is None:
            raise RuntimeError("LB Job ID is null.")

        return self._slurm.get_node_from_jobid(self.lb_jobid)

    def _get_head_node(self) -> str:
        if self.jobid is None:
            raise RuntimeError("Job ID is null.")

        return self._slurm.get_node_from_jobid(self.jobid)

    def _wait_for_lb_health(self):
        """
        Block until *both* the replica-array **and** the Nginx load-balancer
        are up and the LB answers HTTP on /v1/models.  Sets
            • self.lb_node    – host running the LB container
            • self.endpoint   – LB URL (http://host:<lb_port>)
        """
        self._lb_checker.wait_for_lb()

    def submit_job(
        self,
        job: SwarmJob,
        *,
        input_path: Path,
        output_path: Path,
        num_threads: int = 1,
        detach: bool = False,
        limit: int | None = None,
        mail_user: Optional[str] = None,
    ) -> int | None:
        """
        Launch a serialized :class:`~domyn_swarm.SwarmJob` inside the current
        SLURM swarm allocation.

        The *job* object is converted to keyword arguments via
        :py:meth:`SwarmJob.to_kwargs`, transmitted to the head node
        (where ``SLURM_NODEID == 0``), reconstructed by
        ``domyn_swarm.jobs.run``, and executed under ``srun``.

        Parameters
        ----------
        job : SwarmJob
            The job instance to execute.
        input_path : utils.EnvPath | str
            Parquet file produced by the upstream pipeline stage.
        output_path : utils.EnvPath | str
            Destination Parquet file to be written by *job*.
        num_threads : int, default 1
            Number of CPU threads the job may use in the worker process.
        detach : bool, default False
            If *True*, start the job in a new process group and return
            immediately with its PID; if *False* (default) the call blocks
            until completion.
        limit : int or None, optional
            Maximum number of rows to read from *input_path* — handy for
            dry-runs and debugging.  When *None* (default) the entire
            dataset is processed.

        Returns
        -------
        int or None
            *detach=True*  → PID of the spawned process.
            *detach=False* → ``None`` (the call blocks).

        Raises
        ------
        RuntimeError
            The swarm manager is not ready (`self.jobid` or `self.endpoint`
            is ``None``).
        FileNotFoundError
            *input_path* does not exist.
        subprocess.CalledProcessError
            Propagated when the synchronous ``srun`` command exits with a
            non-zero status code.

        Notes
        -----
        The constructed command is logged with *rich* for transparency, e.g.::

            srun --jobid=<...> --nodelist=<...> --ntasks=1 --overlap ...
                python -m domyn_swarm.jobs.run --job-class=<module:Class> ...

        Examples
        --------
        >>> swarm.submit_job(
        ...     my_job,
        ...     input_path=Path("batch.parquet"),
        ...     output_path=Path("predictions.parquet"),
        ...     num_threads=4
        ... )
        """
        input_parquet = to_path(input_path)
        output_parquet = to_path(output_path)

        if self.jobid is None or self.endpoint is None:
            raise RuntimeError("Swarm not ready")

        if self.lb_jobid is None:
            raise RuntimeError("LB Job ID is null.")

        if self.lb_node is None:
            raise RuntimeError("LB Node is null.")

        if not input_parquet.is_file():
            raise FileNotFoundError(input_parquet)

        job_class = f"{job.__class__.__module__}:{job.__class__.__qualname__}"
        job_kwargs = json.dumps(job.to_kwargs())

        if self.cfg.venv_path and self.cfg.venv_path.is_dir():
            python_interpreter = self.cfg.venv_path / "bin" / "python"
        else:
            python_interpreter = sys.executable

        builder = SrunCommandBuilder(self.cfg, self.lb_jobid, self.lb_node).with_env(
            {
                "ENDPOINT": self.endpoint,
                "MODEL": self.model,
                "JOB_CLASS": job_class,
                "JOB_KWARGS": job_kwargs,
            }
        )

        if mail_user is not None:
            self.cfg.mail_user = mail_user
        if self.cfg.mail_user is not None:
            builder = builder.with_mail(self.cfg.mail_user)

        exe = [
            str(python_interpreter),
            "-m",
            "domyn_swarm.jobs.run",
            f"--job-class={job_class}",
            f"--model={self.model}",
            f"--input-parquet={input_parquet}",
            f"--output-parquet={output_parquet}",
            f"--endpoint={self.endpoint}",
            f"--nthreads={num_threads}",
            "--job-kwargs",
            job_kwargs,
        ]

        if limit:
            exe.append(f"--limit={limit}")

        cmd = builder.build(exe)

        logger.info(f"Submitting job {job.__class__.__name__} to swarm {self.jobid}:")

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
                stdout=sys.stdout,
                stderr=sys.stderr,
                start_new_session=True,
                close_fds=True,
            )
            logger.info(f"Detached process with PID {proc.pid}")
            return proc.pid
        else:
            subprocess.run(cmd, check=True, stdout=sys.stdout, stderr=sys.stderr)

    @classmethod
    def from_state(cls, state_file: Path) -> "DomynLLMSwarm":
        """
        Load a swarm from a saved state file (swarm_*.json).
        """
        return SwarmStateManager.load(state_file)

    def _load_state(
        self,
        jobid: int,
        lb_jobid: int,
        lb_node: Optional[str] = None,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
    ) -> "DomynLLMSwarm":
        """
        Load the state of an existing swarm from a state file.
        """
        self.jobid = jobid
        self.lb_jobid = lb_jobid
        self.lb_node = lb_node
        self.endpoint = endpoint
        self.model = model or self.cfg.model

        logger.info(f"Loaded state from {self.jobid} and {self.lb_jobid}")
        return self

    def cleanup(self):
        subprocess.run(["scancel", str(self.jobid)], check=False)
        subprocess.run(["scancel", str(self.lb_jobid)], check=False)
        logger.info(f"scancel {self.jobid} and {self.lb_jobid} requested")
        self._cleaned = True


def _load_job(job_class: str, kwargs_json: str, **kwargs) -> SwarmJob:
    mod, cls = job_class.split(":", 1)
    JobCls = getattr(importlib.import_module(mod), cls)
    return JobCls(**kwargs, **json.loads(kwargs_json))


def _start_swarm(
    name: Optional[str],
    cfg: "DomynLLMSwarmConfig",
    *,
    reverse_proxy: bool = False,
) -> None:
    """Common context-manager + reverse proxy logic."""
    with DomynLLMSwarm(cfg=cfg, name=name) as swarm:
        if reverse_proxy:
            from domyn_swarm.helpers.reverse_proxy import launch_reverse_proxy

            # TODO check this
            if not isinstance(cfg.nginx_image, utils.EnvPath):
                raise RuntimeError("Nginx image is not an env path.")

            if swarm.lb_node is None:
                raise RuntimeError("LB Node is null.")

            if swarm.endpoint is None:
                raise RuntimeError("Endpoint is null.")

            launch_reverse_proxy(
                cfg.nginx_template_path,
                cfg.nginx_image,
                swarm.lb_node,
                swarm._get_head_node(),
                int(swarm.endpoint.split(":")[2]),
                cfg.ray_dashboard_port,
            )
