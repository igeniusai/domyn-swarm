import os
import pathlib
from typing import Optional
from pydantic import BaseModel, ValidationInfo, field_validator, Field
import typer
import yaml

from domyn_swarm import utils
from domyn_swarm.helpers import is_folder, path_exists, to_path
from domyn_swarm.models.driver import DriverConfig
from rich import print as rprint


class DomynLLMSwarmConfig(BaseModel):
    hf_home: utils.EnvPath = utils.EnvPath("/leonardo_work/iGen_train/shared_hf_cache/")

    # model / revision --------------------------------------------------------
    model: str
    revision: str | None = None

    # resources ---------------------------------------------------------------
    replicas: int = 1  # number of cluster replicas (vLLM servers)
    nodes: int = 4  # number of *worker* nodes on each replica (vLLM)
    gpus_per_node: int = 4
    cpus_per_task: int = 8
    mem_per_cpu: str = "40G"
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
        default_factory=lambda: utils.EnvPath(__file__).with_suffix("").parent
        / "templates"
        / "llm_swarm.sh.j2"
    )
    nginx_template_path: utils.EnvPath = Field(
        default_factory=lambda: utils.EnvPath(__file__).with_suffix("").parent
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
                f"[yellow] Huggingface model {v} will be used, make sure that HF_HOME is specified correctly and the model is available in {hf_home}/hub"
            )
        return v


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
