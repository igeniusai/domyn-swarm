# Copyright 2025 iGenius S.p.A
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import math
from pathlib import Path
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    StringConstraints,
    field_validator,
    model_validator,
)
import yaml

from domyn_swarm import utils
from domyn_swarm.config.backend import BackendConfig
from domyn_swarm.config.defaults import default_for
from domyn_swarm.config.plan import DeploymentPlan
from domyn_swarm.config.settings import get_settings
from domyn_swarm.config.watchdog import WatchdogConfig
from domyn_swarm.helpers.io import to_path
from domyn_swarm.helpers.logger import setup_logger

logger = setup_logger(__name__)


class DomynLLMSwarmConfig(BaseModel):
    # model / revision --------------------------------------------------------
    model: str
    name: Annotated[
        str,
        StringConstraints(strip_whitespace=True, to_lower=True, max_length=38),
    ]
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
        default_factory=lambda: utils.EnvPath(get_settings().home),
        description="Home directory where logs and state are stored",
    )

    backend: BackendConfig | None = Field(
        description="Backend configuration for the swarm",
    )
    _plan: DeploymentPlan | None = PrivateAttr(default=None)

    env: dict[str, str] | None = None
    watchdog: WatchdogConfig = Field(default_factory=WatchdogConfig)

    @model_validator(mode="after")
    def _resolve_platform_from_backends(self):
        """
        If `backends` is provided, set a runtime plan now.
        This keeps BC: legacy configs with no `backends` continue to use `platform`.
        """
        if self.backend:
            # Build deployment plans with `self` as context
            # (to access replicas, hf_home, vllm args, etc.)
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

    def persist(self, path: str | Path) -> None:
        config_path = to_path(path)
        with config_path.open("w") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f)

    @model_validator(mode="before")
    @classmethod
    def validate_resource_allocations(cls, data: Any) -> "DomynLLMSwarmConfig":
        """Validate and auto-compute all derived resource allocation fields."""
        replicas = data.get("replicas", 1)
        gpus_per_replica = data.get("gpus_per_replica", 4)
        gpus_per_node = data.get("gpus_per_node", 4)
        replicas_per_node = data.get("replicas_per_node")

        # Replicas per node
        if replicas_per_node is None:
            if gpus_per_replica <= gpus_per_node:
                capacity = gpus_per_node // gpus_per_replica
                replicas_per_node = min(capacity, replicas)
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
            cpus_per_task = max(1, 32 // replicas_per_node) if replicas_per_node else 32

        # Requires Ray?
        requires_ray = gpus_per_replica > gpus_per_node and nodes > 1
        # Ensure watchdog config exists and update ray settings
        if "watchdog" not in data:
            data["watchdog"] = {}
        if "ray" not in data["watchdog"]:
            data["watchdog"]["ray"] = {}
        data["watchdog"]["ray"]["enabled"] = requires_ray

        if requires_ray and gpus_per_replica % gpus_per_node != 0:
            raise ValueError(
                "When gpus_per_replica > gpus_per_node, gpus_per_replica "
                "must be a multiple of gpus_per_node"
            )

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
) -> DomynLLMSwarmConfig:
    """Load YAML, inject driver_script if given, apply replicas override."""
    cfg_dict = yaml.safe_load(config_file)
    cfg = DomynLLMSwarmConfig.model_validate(cfg_dict)
    # override default only if user passed something truthy
    if replicas:
        cfg.replicas = replicas
    return cfg
