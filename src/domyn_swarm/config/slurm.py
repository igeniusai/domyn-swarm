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
    """Configuration for the Nginx load-balancer job fronting the replicas."""

    cpus_per_task: int = Field(
        default=32,
        description=("vCPUs for the driver process that launches and monitors the swarm."),
    )
    mem: str = Field(
        default="16GB",
        description="Physical memory for the driver job.",
    )
    threads_per_core: int = Field(
        default=1,
        description="SMT threads to request per physical core.",
    )
    wall_time: str = Field(
        default="24:00:00",
        description="Slurm time limit for the driver job.",
    )
    enable_proxy_buffering: bool = Field(
        default=True,
        description=(
            "Enable Nginx response and request buffering in the generated load "
            "balancer config. Turn it off for streaming responses that should reach "
            "the client as they are produced."
        ),
    )
    nginx_timeout: str | int = Field(
        default="60s",
        description=(
            "HTTP timeout for the load balancer's proxied requests to the model "
            "replicas. Applied to Nginx's connect, send and read timeouts."
        ),
    )
    port: int = Field(
        default=9000,
        description="External port exposed by the Nginx load balancer.",
    )
    nginx_image: str | utils.EnvPath = Field(
        default_factory=default_for("slurm.endpoint.nginx_image"),
        description=(
            "Path to a Singularity image running Nginx as the swarm's load "
            "balancer. Required on the Slurm backend."
        ),
    )
    qos: str | None = Field(
        default=None,
        description=(
            "QoS override for the load-balancer job. When unset it stays `None`, and "
            "the effective QoS is resolved at submission time as `endpoint.qos` or "
            "`SlurmConfig.qos`; it is not a resolved copy of `SlurmConfig.qos`."
        ),
    )
    poll_interval: int = Field(
        default=10,
        description=(
            "Seconds between `sacct` status checks while waiting for the load "
            "balancer to become ready."
        ),
    )
    require_allocated_node: bool = Field(
        default=False,
        description=(
            "Refuse to build an `srun` command unless already inside a Slurm "
            "allocation. Guards against large data jobs accidentally running on the "
            "load-balancer node."
        ),
    )
    monitoring: MonitoringConfig = Field(
        default_factory=MonitoringConfig,
        description=(
            "Prometheus and GPU-exporter sidecars running alongside the load "
            "balancer. Off by default, and Slurm-only."
        ),
    )


class SlurmConfig(BaseModel):
    """Configuration for SLURM-based deployments."""

    type: Literal["slurm"] = Field(
        default="slurm",
        description="Backend discriminator; always `slurm` for this model.",
    )
    partition: str = Field(
        default_factory=default_for("slurm.partition"),
        description="Slurm partition to submit to.",
    )
    account: str = Field(
        default_factory=default_for("slurm.account"),
        description="Slurm account or charge code.",
    )
    qos: str = Field(
        default_factory=default_for("slurm.qos"),
        description=(
            "Slurm QoS for the cluster and load-balancer jobs. The load balancer "
            "can override it with `endpoint.qos`."
        ),
    )

    # Ray-related settings
    requires_ray: bool | None = Field(
        description="Whether to use Ray for distributed execution",
        default=None,
    )
    ray_port: int = Field(
        default=6379,
        description="Port for Ray's GCS / head node inside each replica.",
    )
    ray_dashboard_port: int = Field(
        default=8265,
        description="Port for the optional Ray dashboard.",
    )

    modules: list[str] = Field(
        default_factory=list,
        description=(
            "Environment modules to `module load` at the top of the generated "
            "cluster sbatch script."
        ),
    )
    preamble: list[str] = Field(
        default_factory=list,
        description=(
            "Additional lines inserted near the top of the generated cluster sbatch "
            "script, before the module loads. Use for extra sbatch directives or "
            "shell setup."
        ),
    )

    template_path: utils.EnvPath = Field(
        default_factory=lambda: (
            utils.EnvPath(__file__).with_suffix("").parent.parent / "templates" / "llm_swarm.sh.j2"
        ),
        description=(
            "Path to the Jinja2 template for the cluster sbatch script. Auto-filled; "
            "there is normally no need to set it."
        ),
    )
    nginx_template_path: utils.EnvPath = Field(
        default_factory=lambda: (
            utils.EnvPath(__file__).with_suffix("").parent.parent / "templates" / "nginx.conf.j2"
        ),
        description=(
            "Path to the Jinja2 template for the Nginx config. Auto-filled; there is "
            "normally no need to set it."
        ),
    )

    time_limit: str = Field(
        default="36:00:00",
        description="Overall Slurm wall-clock limit for the allocation.",
    )
    exclude_nodes: str | None = Field(
        default=None,
        description=("Nodes to exclude, passed through to Slurm, e.g. `node[001-004]`."),
    )
    node_list: str | None = Field(
        default=None,
        description="Explicit node list to run on, e.g. `node[005-008]`.",
    )
    mail_user: str | None = Field(
        default=None,
        description=(
            "Email address for Slurm END and FAIL notifications about the resources "
            "domyn-swarm deploys. Notifications are disabled when unset."
        ),
    )
    endpoint: SlurmEndpointConfig = Field(
        default_factory=SlurmEndpointConfig,
        description="Configuration for the Nginx load-balancer job.",
    )

    venv_path: utils.EnvPath | None = Field(
        default=None,
        description=("Virtual environment used by the driver process, not by the containers."),
    )
    env: dict[str, str] | None = Field(
        default=None,
        description=("Additional environment variables set on every job this backend submits."),
    )

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
