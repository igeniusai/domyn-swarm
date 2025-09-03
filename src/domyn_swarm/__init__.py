import importlib
import json
import logging
import secrets
import string
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

from leptonai.api.v1.types.deployment import (
    EnvVar,
    LeptonContainer,
    LeptonDeploymentUserSpec,
    LeptonResourceAffinity,
    ResourceRequirement,
    TokenVar,
)
from leptonai.api.v1.types.job import LeptonJobUserSpec
from pydantic import BaseModel, Field, ValidationInfo, computed_field, field_validator

from domyn_swarm.backends.compute.lepton import LeptonComputeBackend
from domyn_swarm.backends.compute.slurm import SlurmComputeBackend
from domyn_swarm.backends.serving.lepton import LeptonServingBackend
from domyn_swarm.backends.serving.slurm import SlurmServingBackend
from domyn_swarm.deploy.deployment import Deployment
from domyn_swarm.helpers.data import (
    generate_swarm_name,
)
from domyn_swarm.helpers.io import to_path
from domyn_swarm.helpers.logger import setup_logger
from domyn_swarm.jobs import SwarmJob
from domyn_swarm.models.swarm import DomynLLMSwarmConfig
from domyn_swarm.platform.protocols import ServingHandle

from .core.slurm_driver import SlurmDriver
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

    name: str | None = Field(default_factory=generate_swarm_name)
    cfg: DomynLLMSwarmConfig
    endpoint: Optional[str] = None  # LB endpoint, set after job submission
    delete_on_exit: Optional[bool] = (
        False  # Delete the resources for this cluster at the end of the job
    )
    serving_handle: Optional[ServingHandle] = (
        None  # ServingHandle, set after deployment
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

    def _ensure_ready(self):
        """
        Ensure the swarm is ready to accept jobs.
        Raises RuntimeError if not ready.
        """
        match self.cfg.platform:
            case "slurm":
                if self.serving_handle is None:
                    raise RuntimeError("Serving handle is null.")
                jobid = self.serving_handle.meta.get("jobid")
                lb_jobid = self.serving_handle.meta.get("lb_jobid")
                lb_node = self.serving_handle.meta.get("lb_node")
                endpoint = self.serving_handle.url
                if not all([jobid, lb_jobid, lb_node, endpoint]):
                    raise RuntimeError(
                        f"Swarm not ready (jobid/lb_jobid/lb_node/endpoint): {jobid}/{lb_jobid}/{lb_node}/{endpoint}"
                    )
            case "lepton":
                if self.serving_handle is None:
                    raise RuntimeError("Serving handle is null.")
                endpoint = self.serving_handle.url
                if not endpoint:
                    raise RuntimeError(f"Swarm not ready (endpoint): {endpoint}")
            case "azureml":
                raise NotImplementedError("AzureML platform is not yet supported.")
            case _:
                raise ValueError(f"Unsupported platform: {self.cfg.platform}")

    def model_post_init(self, __context: Any) -> None:
        """Post-init to set up the deployment backend."""
        self._cleaned = False
        match self.cfg.platform:
            case "slurm":
                slurm = SlurmDriver(self.cfg)
                serving = SlurmServingBackend(driver=slurm, cfg=self.cfg)
                pass
            case "azureml":
                raise NotImplementedError("AzureML platform is not yet supported.")
            case "lepton":
                if self.cfg.lepton is None or self.cfg.lepton.workspace_id is None:
                    raise ValueError(
                        "Lepton configuration is required in order to use Lepton as platform."
                    )
                serving = LeptonServingBackend(workspace=self.cfg.lepton.workspace_id)
            case _:
                raise ValueError(f"Unsupported platform: {self.cfg.platform}")

        self._state_mgr = SwarmStateManager(self)
        self._deployment = Deployment(serving=serving)

    def __enter__(self):
        if not self.name:
            raise RuntimeError("Name is null.")
        assert self._deployment is not None

        spec = self._build_serving_spec()  # per-platform; see helper below
        deployment_name = self._deployment_name()  # sanitize/truncate once

        logger.info(f"Creating deployment {deployment_name} on {self.cfg.platform}...")

        handle = self._deployment.up(deployment_name, spec, timeout_s=self.cfg.lb_wait)
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
        self._ensure_ready()
        input_parquet = to_path(input_path)
        output_parquet = to_path(output_path)

        job_class = f"{job.__class__.__module__}:{job.__class__.__qualname__}"
        job_kwargs = json.dumps(job.to_kwargs())

        if (
            self.cfg.platform == "slurm"
            and self.cfg.venv_path
            and self.cfg.venv_path.is_dir()
        ):
            python_interpreter = self.cfg.venv_path / "bin" / "python"
        elif self.cfg.platform == "lepton":
            python_interpreter = "python"
        else:
            python_interpreter = sys.executable

        if mail_user is not None:
            self.cfg.mail_user = mail_user

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

        if self.cfg.env:
            env.update(self.cfg.env)

        image = None
        resources = None
        match self.cfg.platform:
            case "slurm":
                pass
            case "azureml":
                raise NotImplementedError("AzureML platform is not yet supported.")
            case "lepton":
                env["API_TOKEN_SECRET_NAME"] = (
                    self.serving_handle.meta.get("token_secret_name", "")
                    if self.serving_handle
                    else ""
                )
                image = self.cfg.lepton.job.image if self.cfg.lepton else None
                affinity = LeptonResourceAffinity(
                    allowed_dedicated_node_groups=[self.cfg.lepton.job.node_group]
                    if self.cfg.lepton and self.cfg.lepton.job.node_group
                    else None,
                    allowed_nodes_in_node_group=self.cfg.lepton.job.allowed_nodes
                    if self.cfg.lepton and self.cfg.lepton.job.allowed_nodes
                    else None,
                )
                resources = LeptonJobUserSpec(
                    affinity=affinity,
                    resource_shape=self.cfg.lepton.job.resource_shape
                    if self.cfg.lepton
                    else None,
                    completions=1,
                    parallelism=1,
                    mounts=self.cfg.lepton.job.mounts if self.cfg.lepton else None,
                ).model_dump(by_alias=True)

        logger.info(
            f"Submitting job {job.__class__.__name__} to swarm {self.name} on {self.cfg.platform}:"
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

    @classmethod
    def from_state(cls, state_file: Path) -> "DomynLLMSwarm":
        """
        Load a swarm from a saved state file (swarm_*.json).
        """
        return SwarmStateManager.load(state_file)

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
        # per-platform constraints (avoid textwrap.shorten which adds a placeholder)
        if self.cfg.platform == "lepton":
            maxlen = 36
            # only letters, digits, dash/underscore; strip others
            cleaned = "".join(ch for ch in raw if ch.isalnum() or ch in "-_")
            return cleaned[:maxlen] or "swarm"
        elif self.cfg.platform == "slurm":
            # Slurm is lenient, but keep it tidy
            cleaned = "".join(ch for ch in raw if ch.isalnum() or ch in "-_")
            return cleaned[:80] or "swarm"
        else:
            cleaned = "".join(ch for ch in raw if ch.isalnum() or ch in "-_")
            return cleaned[:63] or "swarm"

    def _build_serving_spec(self) -> dict:
        match self.cfg.platform:
            case "slurm":
                return {
                    "replicas": self.cfg.replicas,
                    "gpus_per_replica": self.cfg.gpus_per_replica,
                }
            case "lepton":
                # build Lepton spec; DO NOT store the api_token on self (state is persisted)
                if self.cfg.lepton is None or self.cfg.lepton.endpoint is None:
                    raise ValueError(
                        "Lepton endpoint configuration is required in order to use Lepton as platform."
                    )
                api_token = "".join(
                    secrets.choice(string.ascii_letters + string.digits)
                    for _ in range(32)
                )
                container = LeptonContainer(
                    image=str(self.cfg.vllm_image) or "vllm/vllm-openai",
                    command=[
                        "vllm",
                        "serve",
                        self.cfg.model,
                        "--port",
                        str(self.cfg.vllm_port),
                        "--tensor-parallel-size",
                        str(self.cfg.gpus_per_replica),
                        *(self.cfg.vllm_args or "").split(),
                    ],
                )
                spec = LeptonDeploymentUserSpec(
                    container=container,
                    resource_requirement=ResourceRequirement(
                        resource_shape=self.cfg.lepton.endpoint.resource_shape,
                        min_replicas=self.cfg.replicas,
                        max_replicas=self.cfg.replicas,
                        affinity=LeptonResourceAffinity(
                            allowed_dedicated_node_groups=(
                                [self.cfg.lepton.endpoint.node_group]
                                if self.cfg.lepton.endpoint.node_group
                                else None
                            ),
                            allowed_nodes_in_node_group=self.cfg.lepton.endpoint.allowed_nodes
                            or None,
                        ),
                    ),
                    envs=[
                        EnvVar(name=k, value=v) for k, v in (self.cfg.env or {}).items()
                    ]
                    + [EnvVar(name="HF_HOME", value=str(self.cfg.hf_home))],
                    mounts=self.cfg.lepton.endpoint.mounts,
                    api_tokens=[TokenVar(value=api_token)],
                ).model_dump(by_alias=True)
                return spec
            case _:
                return {}

    def _make_compute_backend(self, handle: ServingHandle):
        match self.cfg.platform:
            case "slurm":
                lb_jobid = handle.meta.get("lb_jobid")
                lb_node = handle.meta.get("lb_node")
                if not lb_jobid or not lb_node:
                    raise RuntimeError("LB Job ID/Node missing in Slurm handle.")
                return SlurmComputeBackend(
                    cfg=self.cfg, lb_jobid=lb_jobid, lb_node=lb_node
                )
            case "lepton":
                return LeptonComputeBackend()
            case _:
                raise NotImplementedError(
                    f"Compute backend for {self.cfg.platform} not implemented"
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
