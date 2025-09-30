import io
import math
import os
from typing import Any, Optional

import yaml
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    ValidationInfo,
    field_validator,
    model_validator,
)
from rich import print as rprint

from domyn_swarm import utils
from domyn_swarm.config.backend import BackendConfig
from domyn_swarm.config.defaults import default_for
from domyn_swarm.config.plan import DeploymentPlan
from domyn_swarm.helpers.io import is_folder, path_exists, to_path


class DomynLLMSwarmConfig(BaseModel):
    # model / revision --------------------------------------------------------
    model: str
    revision: str | None = None

    # resources ---------------------------------------------------------------
    replicas: int = 1  # number of cluster replicas (vLLM servers)
    gpus_per_replica: int = 4  # number of GPUs per replica (vLLM)

    gpus_per_node: int = Field(
        description="Number of GPUs per node (vLLM)",
        default=4,
        ge=1,
        le=4,
    )
    replicas_per_node: int | None = Field(
        description="Number of model replicas per node (vLLM)", default=None
    )
    nodes: int | None = Field(
        description="Number of nodes to use for the swarm (vLLM)", default=None
    )
    cpus_per_task: int | None = Field(
        description="Number of CPUs per task (vLLM)",
        ge=1,
        default=None,
    )
    mem_per_cpu: str | None = None
    wait_endpoint_s: int = 1200  # seconds to wait for LB to be ready

    image: str | utils.EnvPath = Field(default_factory=default_for("image"))

    args: str = ""
    port: int = 8000

    home_directory: utils.EnvPath = Field(
        default_factory=lambda: utils.EnvPath(os.path.join(os.getcwd(), ".domyn_swarm"))
    )

    backend: BackendConfig | None = Field(
        description="List of backend configurations",
    )
    _plan: Optional[DeploymentPlan] = PrivateAttr(default=None)

    env: dict[str, str] | None = None

    @model_validator(mode="after")
    def _resolve_platform_from_backends(self):
        """
        If `backends` is provided, set a runtime plan now.
        This keeps BC: legacy configs with no `backends` continue to use `platform`.
        """
        if self.backend:
            # Build deployment plans with `self` as context (to access replicas, hf_home, vllm args, etc.)
            self._plan = self.backend.build(self)
            if not self._plan:
                raise ValueError("At least one backend must be configured")

        return self

    # Convenience accessor
    def get_deployment_plan(self) -> DeploymentPlan | None:
        return self._plan

    @field_validator("backend")
    @classmethod
    def not_empty(cls, v: BackendConfig | None) -> BackendConfig:
        if not v:
            raise ValueError("At least one backend must be configured")
        return v

    @classmethod
    def read(cls, path: str) -> "DomynLLMSwarmConfig":
        config_path = to_path(path)
        return _load_swarm_config(config_path.open())

    @field_validator("model", mode="after")
    @classmethod
    def validate_model(cls, v: str, info: ValidationInfo):
        if path_exists(v) and is_folder(v):
            rprint(f"Model saved to local folder {v} will be used")
        else:
            hf_home = info.data["env"].get("hf_home") if info.data.get("env") else None
            if not hf_home:
                hf_home = os.getenv(
                    "HF_HOME",
                    os.path.join(os.path.expanduser("~"), ".cache/huggingface"),
                )
            rprint(
                f"[yellow]Huggingface model[/yellow] [bold green]{v}[/bold green] [yellow]will be used, make sure that[/yellow] [bold cyan]HF_HOME[/bold cyan] [yellow]is specified correctly and the model is available in[/yellow] {hf_home}/hub"
            )
        return v

    @model_validator(mode="before")
    @classmethod
    def validate_resource_allocations(cls, data: Any) -> "DomynLLMSwarmConfig":
        """Validate and auto-compute all derived resource allocation fields."""
        replicas = data.get("replicas", 1)
        gpus_per_replica = data.get("gpus_per_replica", 4)
        gpus_per_node = data.get("gpus_per_node", 4)
        replicas_per_node = data.get("replicas_per_node")

        # Replicas per node
        if not replicas_per_node:
            if gpus_per_replica <= gpus_per_node:
                replicas_per_node = gpus_per_node // gpus_per_replica
            else:
                replicas_per_node = None

        # Nodes
        if replicas_per_node:
            nodes = math.ceil(replicas / replicas_per_node)
        else:
            nodes = math.ceil((replicas * gpus_per_replica) / gpus_per_node)

        if nodes < 1:
            raise ValueError("Number of nodes must be >= 1")

        # CPUs per task
        cpus_per_task = data.get("cpus_per_task")
        if cpus_per_task is None:
            if replicas_per_node:
                cpus_per_task = max(1, 32 // replicas_per_node)
            else:
                cpus_per_task = 32

        # Requires Ray?
        requires_ray = gpus_per_replica >= gpus_per_node and nodes > 1

        # Fill computed fields
        data["replicas_per_node"] = replicas_per_node
        data["nodes"] = nodes
        data["cpus_per_task"] = cpus_per_task

        # Update backend configurations with computed values
        backend = data.get("backend", [])
        if backend:
            if not isinstance(backend, dict):
                backend = backend.model_dump()
            if backend.get("type") == "slurm" and "requires_ray" not in backend:
                backend["requires_ray"] = requires_ray

        data["backend"] = backend

        return data


def _load_swarm_config(
    config_file: io.TextIOWrapper,
    *,
    replicas: int | None = None,
    platform: str | None = "slurm",
) -> DomynLLMSwarmConfig:
    """Load YAML, inject driver_script if given, apply replicas override."""
    cfg_dict = yaml.safe_load(config_file)
    cfg = DomynLLMSwarmConfig.model_validate(cfg_dict)
    # override default only if user passed something truthy
    if replicas:
        cfg.replicas = replicas
    return cfg
