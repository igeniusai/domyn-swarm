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

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from domyn_swarm import utils
from domyn_swarm.config.defaults import default_for
from domyn_swarm.config.plan import DeploymentPlan
from domyn_swarm.config.settings import get_settings

settings = get_settings()


_DCGM_DEFAULT_IMAGE = "nvcr.io/nvidia/k8s/dcgm-exporter:3.3.5-3.4.1-ubuntu22.04"


class GpuExporterConfig(BaseModel):
    """Optional per-node GPU metrics exporter for replica nodes.

    Disabled by default. `kind` selects the exporter implementation and the
    bundled dashboard vocabulary. See
    docs/superpowers/specs/2026-07-02-gpu-monitoring-design.md.
    """

    enabled: bool = False
    kind: Literal["nvidia_smi", "dcgm"] = "nvidia_smi"
    image: str | None = None
    binary: str | None = None
    port: int = 9835

    def resolved_binary(self, *, mode: str) -> str:
        if self.binary:
            return self.binary
        if self.kind == "nvidia_smi":
            return "nvidia_gpu_exporter"
        raise ValueError("gpu_exporter.binary is required for kind='dcgm' with mode='binary'")

    def resolved_image(self, *, mode: str) -> str | None:
        if self.image:
            return self.image
        if self.kind == "dcgm":
            return _DCGM_DEFAULT_IMAGE
        return None  # nvidia_smi container mode requires an explicit image


class RayMetricsConfig(BaseModel):
    """Optional scraping of Ray's per-node Prometheus metrics (``ray_*``).

    Only effective for Ray multi-node replicas (``requires_ray``). Resolved to
    ``enabled=True`` when monitoring is on and the deployment requires Ray,
    unless explicitly set to ``False``.

    Attributes:
        enabled: Tri-state. ``None`` means auto (True iff monitoring+requires_ray).
        port: Fixed Ray ``--metrics-export-port`` so announce files are stable.
    """

    enabled: bool | None = None
    port: int = 8090


class MonitoringConfig(BaseModel):
    """Optional Prometheus-based monitoring sidecar for the LB node.

    Disabled by default; when disabled the LB behaves exactly as before. See
    docs/superpowers/specs/2026-06-05-vllm-prometheus-monitoring-design.md.

    Attributes:
        enabled: Master switch. When False, all other fields are ignored.
        mode: 'container' (singularity images) or 'binary' (host binaries).
        prometheus_image: Singularity image for Prometheus (mode='container').
        nginx_exporter_image: Singularity image for nginx-prometheus-exporter.
        prometheus_binary: Prometheus binary name/path (mode='binary').
        nginx_exporter_binary: nginx-exporter binary name/path (mode='binary').
        port: Prometheus listen port on the LB node (proxied; not user-facing).
        exporter_port: nginx-exporter metrics port (scraped by Prometheus).
        route_prefix: nginx path prefix Prometheus is served under.
        scrape_interval: Prometheus global scrape interval (e.g. '15s').
        retention: TSDB retention window (e.g. '12h').
    """

    enabled: bool = False
    mode: Literal["container", "binary"] = "container"
    prometheus_image: str | None = Field(
        default_factory=default_for("slurm.endpoint.prometheus_image", None)
    )
    nginx_exporter_image: str | None = Field(
        default_factory=default_for("slurm.endpoint.nginx_exporter_image", None)
    )
    prometheus_binary: str = "prometheus"
    nginx_exporter_binary: str = "nginx-prometheus-exporter"
    port: int = 9090
    exporter_port: int = 9113
    route_prefix: str = "/prometheus"
    scrape_interval: str = "15s"
    retention: str = "12h"
    gpu_exporter: GpuExporterConfig = Field(default_factory=GpuExporterConfig)
    ray_metrics: RayMetricsConfig = Field(default_factory=RayMetricsConfig)

    @field_validator("route_prefix")
    @classmethod
    def _ensure_leading_slash(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("route_prefix must be a non-empty string")
        return v if v.startswith("/") else f"/{v}"

    @model_validator(mode="after")
    def _validate_gpu_exporter_combo(self) -> "MonitoringConfig":
        if not self.gpu_exporter.enabled:
            return self
        if (
            self.mode == "container"
            and self.gpu_exporter.kind == "nvidia_smi"
            and self.gpu_exporter.image is None
        ):
            raise ValueError(
                "nvidia_smi container mode needs an explicit gpu_exporter.image "
                "(build from images/gpu_exporter_nvidia_smi.def) or use mode=binary."
            )
        if self.mode == "binary" and self.gpu_exporter.kind == "dcgm":
            raise ValueError(
                "dcgm exporter is only supported in container mode (the launch "
                "template runs it via `singularity exec`); set mode=container."
            )
        return self


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
    qos: str | None = None
    poll_interval: int = 10  # sacct polling cadence (s)
    require_allocated_node: bool = False  # refuse srun if not inside a Slurm allocation
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)


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
    modules: list[str] = []
    preamble: list[str] = []  # additional SLURM directives

    template_path: utils.EnvPath = Field(
        default_factory=lambda: (
            utils.EnvPath(__file__).with_suffix("").parent.parent / "templates" / "llm_swarm.sh.j2"
        )
    )
    nginx_template_path: utils.EnvPath = Field(
        default_factory=lambda: (
            utils.EnvPath(__file__).with_suffix("").parent.parent / "templates" / "nginx.conf.j2"
        )
    )

    time_limit: str = "36:00:00"  # e.g. 36 hours
    exclude_nodes: str | None = None  # e.g. "node[1-3]" (optional)
    node_list: str | None = None  # e.g. "node[4-6]" (optional)
    mail_user: str | None = None  # Enable email notifications if set
    endpoint: SlurmEndpointConfig = Field(default_factory=SlurmEndpointConfig)

    venv_path: utils.EnvPath | None = None
    env: dict[str, str] | None = None  # Additional env vars for all jobs

    mounts: list[str] = Field(
        default_factory=list,
        description=(
            "Extra Singularity bind mounts for the vLLM containers. Each entry is "
            "either '/path' (bound at the same path inside the container) or "
            "'/host/path:/container/path' (with an optional ':ro'/':rw' suffix). "
            "Appended verbatim to the container's bind list."
        ),
    )

    @field_validator("mounts")
    @classmethod
    def _validate_mounts(cls, value: list[str]) -> list[str]:
        """Validate the format of each bind mount specification.

        Checks that every entry is non-empty, has at most three colon-separated
        segments (source[:dest[:opts]]), and uses an absolute source path. Host
        path existence is intentionally not checked here.

        Args:
            value: The list of mount specifications.

        Returns:
            The validated list of mount specifications.

        Raises:
            ValueError: If any entry is empty, has too many ':' segments, or has
                a non-absolute source path.
        """
        for mount in value:
            if not mount or not mount.strip():
                raise ValueError("mount entry must be a non-empty string")
            segments = mount.split(":")
            if len(segments) > 3:
                raise ValueError(
                    f"invalid mount spec '{mount}': expected at most 'source:dest:opts'"
                )
            if not segments[0].startswith("/"):
                raise ValueError(f"invalid mount spec '{mount}': source must be an absolute path")
        return value

    def build(self, cfg_ctx) -> DeploymentPlan:
        """Builds the deployment plan for SLURM-based deployments."""
        from domyn_swarm.backends.compute.slurm import SlurmComputeBackend
        from domyn_swarm.backends.serving.slurm import SlurmServingBackend
        from domyn_swarm.backends.serving.slurm_driver import SlurmDriver

        driver = SlurmDriver(cfg=cfg_ctx)
        serving = SlurmServingBackend(cfg=self, driver=driver)
        compute = SlurmComputeBackend(cfg=self, lb_jobid=0, lb_node="")

        serving_spec = self.model_dump(exclude_none=True) | cfg_ctx.model_dump(
            include={
                "replicas",
                "nodes",
                "gpus_per_replica",
                "gpus_per_node",
                "replicas_per_node",
            },
            exclude_none=True,
        )

        return DeploymentPlan(
            name_hint="slurm",
            serving=serving,
            compute=compute,
            serving_spec=serving_spec,
            job_resources={},
            extras={},
            platform="slurm",
        )
