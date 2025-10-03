import importlib
import json
import logging
import uuid
import warnings
from pathlib import Path
from typing import Any, Optional

from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    computed_field,
)

from domyn_swarm.backends.compute.slurm import SlurmComputeBackend
from domyn_swarm.config.settings import get_settings
from domyn_swarm.config.slurm import SlurmConfig
from domyn_swarm.config.swarm import DomynLLMSwarmConfig
from domyn_swarm.deploy.deployment import Deployment
from domyn_swarm.helpers.io import to_path
from domyn_swarm.helpers.logger import setup_logger
from domyn_swarm.helpers.swarm import generate_swarm_name
from domyn_swarm.jobs import SwarmJob
from domyn_swarm.platform.protocols import ServingHandle, ServingPhase, ServingStatus

from ..core.state import SwarmStateManager

logger = setup_logger(__name__, level=logging.INFO)


class DomynLLMSwarm(BaseModel):
    """
    A context manager for orchestrating distributed LLM serving swarms across compute platforms.

    This class provides a unified interface for deploying, managing, and interacting with
    large language model serving clusters on various compute backends (Slurm, cloud platforms).
    It handles the complete lifecycle from resource allocation to cleanup, with built-in
    state persistence and job submission capabilities.

    Architecture Overview
    ---------------------
    The swarm consists of multiple components:
    - Load balancer (nginx) for request distribution
    - Multiple vLLM server instances for model serving
    - Head node for job coordination and user script execution
    - State management for persistence and recovery

    Platform Support
    ----------------
    - **Slurm**: Deploys as job arrays with SLURM_NODEID-based role assignment
      - Node 0: Load balancer + user driver
      - Nodes 1-N: vLLM servers
    - **Cloud platforms**: Via deployment abstractions (Lepton, etc.)

    Attributes
    ----------
    cfg : DomynLLMSwarmConfig
        Configuration object containing deployment parameters, resource requirements,
        and platform-specific settings.
    endpoint : str, optional
        The public URL endpoint of the deployed swarm load balancer. Set after
        successful deployment via the context manager.
    delete_on_exit : bool, default False
        Whether to automatically clean up all allocated resources when exiting
        the context manager. Useful for temporary deployments.
    serving_handle : ServingHandle, optional
        Platform-specific handle for the serving deployment, containing metadata
        like job IDs, node assignments, and status information.
    model : str
        The name/path of the LLM being served. Can be set after initialization
        for dynamic model switching scenarios.

    Methods
    -------
    submit_job(job, input_path, output_path, **kwargs)
        Execute a SwarmJob within the allocated cluster resources.
    submit_script(script_path, detach=False, extra_args=None)
        Run arbitrary Python scripts on the head node with swarm environment.
    status()
        Query current deployment status and health information.
    cleanup() / down()
        Manually terminate and clean up all allocated resources.
    from_state(deployment_name)
        Restore a swarm from previously persisted state.
    delete_record(deployment_name)
        Remove deployment records from state storage.

    Usage Patterns
    --------------
    **Basic Deployment:**
    ```python
    cfg = DomynLLMSwarmConfig(...)
    with DomynLLMSwarm(cfg=cfg) as swarm:
        # Swarm is now running and accessible at swarm.endpoint
        result = swarm.submit_job(my_job, input_path="data.parquet",
                                 output_path="results.parquet")
    # Resources automatically cleaned up on exit
    ```

    **Persistent Deployment:**
    ```python
    swarm = DomynLLMSwarm(cfg=cfg, delete_on_exit=False)
    with swarm:
        # Do work...
        pass
    # Resources remain allocated

    # Later, reconnect to existing deployment
    swarm = DomynLLMSwarm.from_state("my-deployment-abc123")
    ```

    **Job Submission:**
    ```python
    with DomynLLMSwarm(cfg=cfg) as swarm:
        # Synchronous execution
        swarm.submit_job(
            job=MyProcessingJob(),
            input_path="batch.parquet",
            output_path="predictions.parquet",
            num_threads=8,
            limit=1000  # Process first 1000 rows only
        )

        # Asynchronous execution
        pid = swarm.submit_job(
            job=LongRunningJob(),
            input_path="large_dataset.parquet",
            output_path="results.parquet",
            detach=True
        )
    ```

    **Script Execution:**
    ```python
    with DomynLLMSwarm(cfg=cfg) as swarm:
        # Run analysis script on head node
        swarm.submit_script(
            Path("analysis.py"),
            extra_args=["--mode", "evaluation"],
            detach=False
        )
    ```

    State Management
    ----------------
    The swarm automatically persists its state including:
    - Deployment metadata and resource handles
    - Configuration parameters
    - Platform-specific identifiers (job IDs, node assignments)
    - Endpoint URLs and connection information

    This enables recovery from failures and reconnection to existing deployments
    across process boundaries.

    Error Handling
    --------------
    - **Resource allocation failures**: Raises RuntimeError with diagnostic info
    - **Timeout during startup**: Controlled by cfg.wait_endpoint_s parameter
    - **Job submission errors**: Propagates subprocess.CalledProcessError
    - **Cleanup failures**: Logged but don't prevent context exit

    Platform-Specific Behavior
    ---------------------------
    **Slurm:**
    - Uses job arrays for multi-node allocation
    - Leverages SLURM_NODEID for role assignment
    - Supports srun-based job execution with resource isolation
    - Integrates with Slurm job lifecycle (PENDING→RUNNING→COMPLETED)

    **Cloud Platforms:**
    - Abstract deployment via platform-specific drivers
    - Resource scaling based on platform capabilities
    - Native load balancing and health monitoring

    Notes
    -----
    - The swarm must be used as a context manager for proper resource lifecycle
    - All paths in job submission are resolved relative to the execution environment
    - Environment variables (ENDPOINT, MODEL) are automatically set for submitted jobs
    - Checkpoint directories are created automatically for job state persistence

    See Also
    --------
    SwarmJob : Base class for jobs executable within the swarm
    DomynLLMSwarmConfig : Configuration schema and validation
    Deployment : Low-level deployment abstraction
    SwarmStateManager : State persistence and recovery system

    Examples
    --------
    >>> from domyn_swarm.config.swarm import DomynLLMSwarmConfig
    >>> cfg = DomynLLMSwarmConfig.from_yaml("config.yaml")
    >>> with DomynLLMSwarm(cfg=cfg) as swarm:
    ...     print(f"Swarm available at: {swarm.endpoint}")
    ...     status = swarm.status()
    ...     print(f"Status: {status.phase}")
    """

    cfg: DomynLLMSwarmConfig
    name: str = Field(
        default_factory=lambda data: data.get(
            "name", generate_swarm_name(data["cfg"].name)
        ),
        description="Unique name for this swarm deployment",
    )
    endpoint: Optional[str] = None  # LB endpoint, set after job submission
    delete_on_exit: Optional[bool] = (
        False  # Delete the resources for this cluster at the end of the job
    )
    serving_handle: Optional[ServingHandle] = (
        None  # ServingHandle, set after deployment
    )
    _platform: str = PrivateAttr("")

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
        assert self._deployment is not None

        serving_spec = dict(self._plan.serving_spec)
        name_hint = self._plan.name_hint

        logger.info(
            f"Creating deployment [cyan]{self.name}[/cyan] on {self._platform} ({name_hint})..."
        )

        handle = self._deployment.up(
            self.name, serving_spec, timeout_s=self.cfg.wait_endpoint_s
        )
        self.serving_handle = handle
        self.endpoint = handle.url
        self._deployment.compute = self._make_compute_backend(handle)

        self._persist(self.name)

        return self

    def __exit__(self, exc_type, exc, tb):
        if self.delete_on_exit:
            self.cleanup()

    def _persist(self, deployment_name: str):
        """Save the state.

        Args:
            deployment_name (str): Deployment name.
        """
        self._state_mgr.save(deployment_name)

    @classmethod
    def from_state(cls, deployment_name: str) -> "DomynLLMSwarm":
        """Initialize a swarm from a saved state.

        Args:
            deployment_name (str): Deployment name.

        Returns:
            DomynLLMSwarm: Loaded swarm.
        """
        return SwarmStateManager.load(deployment_name)

    def delete_record(self, deployment_name: str) -> None:
        """Delete swarm from the DB

        Args:
            deployment_name (str): Deployment name.
        """
        self._state_mgr.delete_record(deployment_name)

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
        image = compute.default_image(self.cfg.backend)

        resources = self._plan.job_resources or compute.default_resources(
            self.cfg.backend
        )
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

        settings = get_settings()
        token = (
            settings.api_token
            or settings.vllm_api_key
            or settings.singularityenv_vllm_api_key
        )

        env = {
            "ENDPOINT": self.endpoint,
            "MODEL": self.model,
            "JOB_CLASS": job_class,
            "JOB_KWARGS": f"'{job_kwargs}'",
        }

        if self.cfg.backend and self.cfg.backend.env:
            env.update(self.cfg.backend.env)
        if env_overrides:
            env.update(env_overrides)
        if token:
            env["DOMYN_SWARM_API_TOKEN"] = token.get_secret_value()
            env["VLLM_API_KEY"] = token.get_secret_value()

        job_name = job.name.lower() if job.name else f"{self.name}-job"
        job_name = f"{self.name}-{job_name}"

        logger.info(
            f"Submitting {job.__class__.__name__} [cyan]{job_name}[/cyan] job to swarm {self.name} on {self._platform}"
        )

        job_handle = self._deployment.run(
            name=job_name[:36],  # type: ignore[arg-type]
            image=image,
            command=exe,
            env=env,
            resources=resources,
            detach=detach,
        )

        if job_handle is None:
            raise RuntimeError("Failed to submit job to compute backend.")

        return job_handle.meta.get("pid") if detach else None

    def _compose_runtime(self, extra_env: dict | None = None):
        """Build interpreter/image/resources/env in one place."""
        self._deployment.ensure_ready()
        assert self._deployment.compute is not None, "Compute backend not initialized"

        compute = self._deployment.compute
        python_interpreter = compute.default_python(self.cfg)
        image = compute.default_image(self.cfg.backend)
        resources = compute.default_resources(self.cfg.backend)

        env = {
            "ENDPOINT": self.endpoint,
            "MODEL": self.model,
        }
        # Global backend env from config
        if getattr(self.cfg, "backend", None) and getattr(
            self.cfg.backend, "env", None
        ):
            env.update(self.cfg.backend.env)  # type: ignore[attr-defined]
        # Backend-specific overrides
        overrides = compute.default_env(self.cfg)
        if overrides:
            env.update(overrides)
        # Call-site extras
        if extra_env:
            env.update(extra_env)

        return str(python_interpreter), image, resources, env

    def submit_script(
        self,
        script_path: Path,
        detach: bool = False,
        extra_args: list[str] | None = None,
    ) -> int | None:
        """Submit a Python script to the compute backend for execution.

        This method validates the script path, composes the runtime environment,
        and submits the script for execution via the configured deployment backend.

        Args:
            script_path (Path): Path to the Python script to be executed.
            detach (bool, optional): If True, run the script in detached mode and return
                the process ID. If False, run synchronously. Defaults to False.
            extra_args (list[str] | None, optional): Additional command-line arguments
                to pass to the script. Defaults to None.

        Returns:
            int | None: If detach is True, returns the process ID of the submitted job.
                If detach is False, returns None after synchronous execution.

        Raises:
            FileNotFoundError: If the script file does not exist (only checked for SLURM platform).
            RuntimeError: If the script submission to the compute backend fails.

        Example:
            >>> swarm = Swarm(...)
            >>> # Submit script synchronously
            >>> swarm.submit_script(Path("my_script.py"))

            >>> # Submit script in detached mode with arguments
            >>> pid = swarm.submit_script(
            ...     Path("my_script.py"),
            ...     detach=True,
            ...     extra_args=["--config", "config.yaml"]
            ... )
        """

        # Basic validation
        if self._platform == "slurm" and not script_path.is_file():
            raise FileNotFoundError(f"Script not found: {script_path}")

        # Compose runtime (interpreter/image/resources/env) once
        python_interpreter, image, resources, env = self._compose_runtime()

        # Build the command
        args = extra_args or []
        command = [python_interpreter, str(script_path), *args]

        # Submit via the compute backend
        job_handle = self._deployment.run(
            name=f"{self.name.lower()}-script",  # type: ignore[arg-type]
            image=image,
            command=command,
            env=env,
            resources=resources,
            detach=detach,
        )
        if job_handle is None:
            raise RuntimeError("Failed to submit script to compute backend.")

        return job_handle.meta.get("pid") if detach else None

    def cleanup(self):
        if self._deployment and self.serving_handle:
            self._deployment.down(self.serving_handle)
        self._cleaned = True

    def down(self):
        """Manually clean up the swarm resources."""
        self.cleanup()

    def _deployment_name(self) -> str:
        unique_id = uuid.uuid4()
        short_id = str(unique_id)[:8]
        return f"{self.cfg.name}-{short_id}"

    def _make_compute_backend(self, handle: ServingHandle):
        if self._plan and self._plan.platform == "slurm":
            lb_jobid = handle.meta.get("lb_jobid")
            lb_node = handle.meta.get("lb_node")
            if not lb_jobid or not lb_node:
                raise RuntimeError("LB Job ID/Node missing in Slurm handle.")
            assert self.cfg.backend is not None and isinstance(
                self.cfg.backend, SlurmConfig
            )
            return SlurmComputeBackend(
                cfg=self.cfg.backend, lb_jobid=lb_jobid, lb_node=lb_node
            )
        elif self._plan and self._plan.platform == "lepton":
            return self._plan.compute
        else:
            raise RuntimeError(
                f"Unsupported platform for compute backend: {self._plan.platform}"
            )

    def status(self) -> ServingStatus:
        """Get the current status of the swarm."""
        if self._deployment:
            s = self._deployment.status()
            if s is not None:
                return s
        return ServingStatus(phase=ServingPhase.UNKNOWN, url=self.endpoint)


def _load_job(job_class: str, kwargs_json: str, **kwargs) -> SwarmJob:
    mod, cls = job_class.split(":", 1)
    JobCls = getattr(importlib.import_module(mod), cls)
    return JobCls(**kwargs, **json.loads(kwargs_json))


def _start_swarm(
    cfg: "DomynLLMSwarmConfig",
    *,
    reverse_proxy: bool = False,
) -> None:
    """Common context-manager + reverse proxy logic."""
    with DomynLLMSwarm(cfg=cfg) as _:
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
