import math
import os
from typing import Any, Optional, Self

import typer
import yaml
from pydantic import (
    BaseModel,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)
from rich import print as rprint

from domyn_swarm import utils
from domyn_swarm.helpers.io import is_folder, path_exists, to_path
from domyn_swarm.models.driver import DriverConfig


class DomynLLMSwarmConfig(BaseModel):
    hf_home: utils.EnvPath = utils.EnvPath("/leonardo_work/iGen_train/shared_hf_cache/")

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
    requires_ray: bool | None = Field(
        description="Whether to use Ray for distributed execution",
        default=None,
    )
    mem_per_cpu: str | None = None
    partition: str = "boost_usr_prod"
    account: str = "iGen_train"

    # container images --------------------------------------------------------
    vllm_image: str | utils.EnvPath = utils.EnvPath(
        "/leonardo_work/iGen_train/fdambro1/images/vllm_0.9.1.sif"
    )
    nginx_image: str | utils.EnvPath = utils.EnvPath(
        "/leonardo_work/iGen_train/fdambro1/images/nginx-dask.sif"
    )
    lb_wait: int = 1200  # seconds to wait for LB to be ready
    lb_port: int = 9000

    home_directory: utils.EnvPath = Field(
        default_factory=lambda: utils.EnvPath(os.path.join(os.getcwd(), ".domyn_swarm"))
    )

    log_directory: Optional[utils.EnvPath] = Field(
        default_factory=lambda data: data["home_directory"] / "logs"
    )

    # misc --------------------------------------------------------------------
    max_concurrent_requests: int = 2_000
    poll_interval: int = 10  # sacct polling cadence (s)

    # template path (auto-filled after clone)
    template_path: utils.EnvPath = Field(
        default_factory=lambda: utils.EnvPath(__file__).with_suffix("").parent.parent
        / "templates"
        / "llm_swarm.sh.j2"
    )
    nginx_template_path: utils.EnvPath = Field(
        default_factory=lambda: utils.EnvPath(__file__).with_suffix("").parent.parent
        / "templates"
        / "nginx.conf.j2"
    )
    vllm_args: str = ""
    vllm_port: int = 8000
    ray_port: int = 6379
    ray_dashboard_port: int = 8265
    venv_path: utils.EnvPath | None = None

    time_limit: str = "36:00:00"  # e.g. 36 hours
    exclude_nodes: str | None = None  # e.g. "node[1-3]" (optional)
    node_list: str | None = None  # e.g. "node[4-6]" (optional)

    # mail notification -------------------------------------------------------
    mail_user: str | None = None  # Enable email notifications if set

    driver: DriverConfig | None = DriverConfig()

    def model_post_init(self, context):
        os.makedirs(self.log_directory, exist_ok=True)
        os.makedirs(self.home_directory, exist_ok=True)
        return super().model_post_init(context)

    @classmethod
    def read(cls, path: str | utils.EnvPath) -> "DomynLLMSwarmConfig":
        path = to_path(path)
        return _load_swarm_config(path.open())

    @field_validator("model", mode="after")
    @classmethod
    def validate_model(cls, v: str, info: ValidationInfo):
        if path_exists(v) and is_folder(v):
            rprint(f"Model saved to local folder {v} will be used")
        else:
            hf_home = info.data["hf_home"]
            rprint(
                f"[yellow]Huggingface model[/yellow] [bold green]{v}[/bold green] [yellow]will be used, make sure that[/yellow] [bold cyan]HF_HOME[/bold cyan] [yellow]is specified correctly and the model is available in[/yellow] {hf_home}/hub"
            )
        return v

    @model_validator(mode="before")
    @classmethod
    def validate_resource_allocations(cls, data: Any) -> Self:
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
        requires_ray = gpus_per_replica > gpus_per_node and nodes > 1

        # Fill computed fields
        data["replicas_per_node"] = replicas_per_node
        data["nodes"] = nodes
        data["cpus_per_task"] = cpus_per_task
        data["requires_ray"] = requires_ray

        return data


def _load_swarm_config(
    config_file: typer.FileText, *, replicas: int | None = None
) -> DomynLLMSwarmConfig:
    """Load YAML, inject driver_script if given, apply replicas override."""
    cfg_dict = yaml.safe_load(config_file)
    cfg = DomynLLMSwarmConfig(**cfg_dict)
    # override default only if user passed something truthy
    if replicas:
        cfg.replicas = replicas
    return cfg


# hf download epfml/FineWeb2-HQ --repo-type dataset --local-dir $WORK/datasets/fineweb-2-hq --max-workers 16
