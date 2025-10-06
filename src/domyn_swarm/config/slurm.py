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

import os
from typing import Literal

from pydantic import BaseModel, Field

from domyn_swarm import utils
from domyn_swarm.backends.serving.slurm_driver import SlurmDriver
from domyn_swarm.config.defaults import default_for
from domyn_swarm.config.plan import DeploymentPlan
from domyn_swarm.config.settings import get_settings

settings = get_settings()


class SlurmEndpointConfig(BaseModel):
    cpus_per_task: int = 32
    mem: str = "16GB"
    threads_per_core: int = 1
    wall_time: str = "24:00:00"
    enable_proxy_buffering: bool = True
    nginx_timeout: str | int = "60s"
    port: int = 9000
    nginx_image: str | utils.EnvPath = Field(
        default_factory=default_for("slurm.endpoint.nginx_image")
    )
    poll_interval: int = 10  # sacct polling cadence (s)


class SlurmConfig(BaseModel):
    """Configuration for SLURM-based deployments."""

    type: Literal["slurm"] = "slurm"
    partition: str = Field(default_factory=default_for("slurm.partition"))
    account: str = Field(default_factory=default_for("slurm.account"))
    qos: str = Field(default_factory=default_for("slurm.qos"))

    # Ray-related settings
    requires_ray: bool | None = Field(
        description="Whether to use Ray for distributed execution",
        default=None,
    )
    ray_port: int = 6379
    ray_dashboard_port: int = 8265

    # Additional SLURM settings, not yet exposed and used anywhere
    module_load: list[str] = []
    preamble: list[str] = []  # additional SLURM directives

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

    time_limit: str = "36:00:00"  # e.g. 36 hours
    exclude_nodes: str | None = None  # e.g. "node[1-3]" (optional)
    node_list: str | None = None  # e.g. "node[4-6]" (optional)
    mail_user: str | None = None  # Enable email notifications if set
    endpoint: SlurmEndpointConfig = Field(default_factory=SlurmEndpointConfig)

    home_directory: utils.EnvPath = Field(default=utils.EnvPath(get_settings().home))

    log_directory: utils.EnvPath = Field(
        default_factory=lambda data: data["home_directory"] / "logs"
    )
    venv_path: utils.EnvPath | None = None
    env: dict[str, str] | None = None  # Additional env vars for all jobs

    def model_post_init(self, context):
        os.makedirs(self.log_directory, exist_ok=True)
        return super().model_post_init(context)

    def build(self, cfg_ctx) -> DeploymentPlan:
        """Builds the deployment plan for SLURM-based deployments."""
        from domyn_swarm.backends.compute.slurm import SlurmComputeBackend
        from domyn_swarm.backends.serving.slurm import SlurmServingBackend

        driver = SlurmDriver(cfg=cfg_ctx)
        serving = SlurmServingBackend(driver=driver, cfg=self)
        compute = SlurmComputeBackend(cfg=self, lb_jobid=0, lb_node="")

        return DeploymentPlan(
            name_hint="slurm",
            serving=serving,
            compute=compute,
            serving_spec=self.model_dump(exclude_none=True),
            job_resources={},
            extras={},
            platform="slurm",
        )
