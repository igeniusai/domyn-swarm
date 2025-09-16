import importlib
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Optional

from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    ValidationInfo,
    computed_field,
    field_validator,
)

from domyn_swarm.backends.compute.slurm import SlurmComputeBackend
from domyn_swarm.config.slurm import SlurmConfig
from domyn_swarm.config.swarm import DomynLLMSwarmConfig
from domyn_swarm.deploy.deployment import Deployment
from domyn_swarm.helpers.data import (
    generate_swarm_name,
)
from domyn_swarm.helpers.io import to_path
from domyn_swarm.helpers.logger import setup_logger
from domyn_swarm.jobs import SwarmJob
from domyn_swarm.platform.protocols import ServingHandle

from ..core.state import SwarmStateManager

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

    name: str | None = Field(default_factory=generate_swarm_name)
    cfg: DomynLLMSwarmConfig
    endpoint: Optional[str] = None  # LB endpoint, set after job submission
    delete_on_exit: Optional[bool] = (
        False  # Delete the resources for this cluster at the end of the job
    )
    serving_handle: Optional[ServingHandle] = (
        None  # ServingHandle, set after deployment
    )
    _platform: str = PrivateAttr("")

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
        """Post-init to set up the deployment backend."""
        self._cleaned = False
        self._state_mgr = SwarmStateManager(self)

        plan = self.cfg.get_deployment_plan()
        if plan is not None:
            self._plan = plan
            self._platform = plan.platform
            self._deployment = Deployment(
                serving=plan.serving, compute=plan.compute, extras=plan.extras
            )
            return

    def __enter__(self):
        if not self.name:
            raise RuntimeError("Name is null.")
        assert self._deployment is not None

        serving_spec = dict(self._plan.serving_spec)
        name_hint = self._plan.name_hint

        deployment_name = self._deployment_name()

        logger.info(
            f"Creating deployment {deployment_name} on {self._platform} ({name_hint})..."
        )

        handle = self._deployment.up(
            deployment_name, serving_spec, timeout_s=self.cfg.wait_endpoint_s
        )
        self.serving_handle = handle
        self.endpoint = handle.url
        self._deployment.compute = self._make_compute_backend(handle)

        self._persist()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.delete_on_exit:
            self.cleanup()

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
        pass

        # if self.lb_jobid is None:
        #     raise RuntimeError("LB Job ID is null.")

        # if self.lb_node is None:
        #     raise RuntimeError("LB Node is null.")

        # if self.endpoint is None:
        #     raise RuntimeError("Endpoint is null.")

        # if self.cfg.venv_path is None:
        #     raise RuntimeError("Venv path is None.")

        # if self.jobid is None:
        #     raise RuntimeError("No job submitted yet")
        # if not script_path.is_file():
        #     raise FileNotFoundError(f"Script not found: {script_path}")

        # logger.info(f"Submitting user script {script_path} to job {self.jobid}")

        # builder = SrunCommandBuilder(self.cfg, self.lb_jobid, self.lb_node).with_env(
        #     {"ENDPOINT": self.endpoint, "MODEL": self.model}
        # )

        # if self.cfg.mail_user:
        #     builder = builder.with_mail(self.cfg.mail_user)

        # extra = [] if extra_args is None else extra_args
        # cmd = builder.build(
        #     [
        #         str(self.cfg.venv_path / "bin" / "python"),
        #         str(script_path),
        #         *extra,
        #     ]
        # )

        # if detach:
        #     proc = subprocess.Popen(
        #         cmd,
        #         stdout=sys.stdout,
        #         stderr=sys.stderr,
        #         start_new_session=True,
        #         close_fds=True,
        #     )
        #     logger.info(f"Detached process with PID {proc.pid}")
        #     return proc.pid
        # else:
        #     subprocess.run(
        #         cmd,
        #         check=True,
        #         stdout=sys.stdout,
        #         stderr=sys.stderr,
        #     )

    def _persist(self):
        self._state_mgr.save()

    @classmethod
    def from_state(cls, jobid: int, home_directory: Path) -> "DomynLLMSwarm":
        """Initialize a swarm from a saved state.

        Args:
            jobid (int): Job id.
            home_directory (Path): Domyn-swarm home directory.

        Returns:
            DomynLLMSwarm: Loaded swarm.
        """
        return SwarmStateManager.load(jobid, home_directory)

    def delete_record(self) -> None:
        """Delete swarm from the DB"""
        self._state_mgr.delete_record()

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
        checkpoint_dir: str | Path = ".checkpoints",
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
        self._deployment.ensure_ready()
        input_parquet = to_path(input_path)
        output_parquet = to_path(output_path)

        job_class = f"{job.__class__.__module__}:{job.__class__.__qualname__}"
        job_kwargs = json.dumps(job.to_kwargs())

        compute = self._deployment.compute
        assert compute is not None, "Compute backend not initialized"
        python_interpreter = compute.default_python(self.cfg)
        image = compute.default_image(self.cfg)
        resources = compute.default_resources(self.cfg)
        env_overrides = compute.default_env(self.cfg)

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
            f"--checkpoint-dir={checkpoint_dir}",
            "--job-kwargs",
            job_kwargs,
        ]

        if limit:
            exe.append(f"--limit={limit}")

        env = {
            "ENDPOINT": self.endpoint,
            "MODEL": self.model,
            "JOB_CLASS": job_class,
            "JOB_KWARGS": job_kwargs,
        }

        if self.cfg._backend_config and self.cfg._backend_config.env:
            env.update(self.cfg._backend_config.env)
        if env_overrides:
            env.update(env_overrides)

        logger.info(
            f"Submitting job {job.__class__.__name__} to swarm {self.name} on {self._platform}:"
        )

        job_handle = self._deployment.run(
            name=f"{self.name.lower()}-job",  # type: ignore[arg-type]
            image=image,
            command=exe,
            env=env,
            resources=resources,
            detach=detach,
        )

        if job_handle is None:
            raise RuntimeError("Failed to submit job to compute backend.")

        return job_handle.meta.get("pid") if detach else None

    def cleanup(self):
        if self._deployment and self.serving_handle:
            self._deployment.down(self.serving_handle)
        self._cleaned = True

    def down(self):
        """Manually clean up the swarm resources."""
        self.cleanup()

    def _deployment_name(self) -> str:
        assert self.name is not None
        raw = self.name.lower()
        # per-platform constraints
        if self._platform == "lepton":
            maxlen = 36
            # only letters, digits, dash/underscore; strip others
            cleaned = "".join(ch for ch in raw if ch.isalnum() or ch in "-_")
            return cleaned[:maxlen] or "swarm"
        elif self._platform == "slurm":
            # Slurm is lenient, but keep it tidy
            cleaned = "".join(ch for ch in raw if ch.isalnum() or ch in "-_")
            return cleaned[:80] or "swarm"
        else:
            cleaned = "".join(ch for ch in raw if ch.isalnum() or ch in "-_")
            return cleaned[:63] or "swarm"

    def _make_compute_backend(self, handle: ServingHandle):
        if self._plan and self._plan.platform == "slurm":
            lb_jobid = handle.meta.get("lb_jobid")
            lb_node = handle.meta.get("lb_node")
            if not lb_jobid or not lb_node:
                raise RuntimeError("LB Job ID/Node missing in Slurm handle.")
            assert self.cfg._backend_config is not None and isinstance(
                self.cfg._backend_config, SlurmConfig
            )
            return SlurmComputeBackend(
                cfg=self.cfg._backend_config, lb_jobid=lb_jobid, lb_node=lb_node
            )
        elif self._plan and self._plan.platform == "lepton":
            return self._plan.compute
        else:
            raise RuntimeError(
                f"Unsupported platform for compute backend: {self._plan.platform}"
            )


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
    with DomynLLMSwarm(cfg=cfg, name=name) as _:
        if reverse_proxy:
            warnings.warn("This feature is not currently enabled")
            # from domyn_swarm.helpers.reverse_proxy import launch_reverse_proxy

            # # TODO check this
            # if not isinstance(cfg.nginx_image, utils.EnvPath):
            #     raise RuntimeError("Nginx image is not an env path.")

            # if swarm.lb_node is None:
            #     raise RuntimeError("LB Node is null.")

            # if swarm.endpoint is None:
            #     raise RuntimeError("Endpoint is null.")

            # launch_reverse_proxy(
            #     cfg.nginx_template_path,
            #     cfg.nginx_image,
            #     swarm.lb_node,
            #     swarm._get_head_node(),
            #     int(swarm.endpoint.split(":")[2]),
            #     cfg.ray_dashboard_port,
            # )
