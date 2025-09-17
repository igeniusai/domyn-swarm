import secrets
import string
from typing import Literal

from leptonai.api.v1.types.deployment import (
    EnvVar,
    LeptonContainer,
    LeptonDeploymentUserSpec,
    LeptonResourceAffinity,
    Mount,
    ResourceRequirement,
    TokenVar,
)
from leptonai.api.v1.types.job import LeptonJobUserSpec
from pydantic import BaseModel, Field

from domyn_swarm.config.plan import DeploymentPlan

_DEFAULT_RESOURCE_SHAPES = {
    8: "gpu.8xh200",
    4: "gpu.4xh200",
    2: "gpu.2xh200",
    1: "gpu.1xh200",
}


class LeptonEndpointConfig(BaseModel):
    image: str = "vllm/vllm-openai:latest"
    allowed_dedicated_node_groups: list[str] | None = None
    resource_shape: str = "gpu.8xh200"
    allowed_nodes: list[str] = Field(default_factory=list)
    mounts: list[Mount] = [
        Mount.model_validate(
            {
                "path": "/",
                "from": "node-nfs:lepton-shared-fs",
                "mount_path": "/mnt/lepton-shared-fs",
                "mount_options": {"local_cache_size_mib": None, "read_only": None},
            }
        )
    ]
    env: dict[str, str] = Field(default_factory=dict)
    api_token_secret_name: str | None = None


class LeptonJobConfig(BaseModel):
    allowed_dedicated_node_groups: list[str] | None = None
    image: str = "igenius/domyn-swarm:latest"
    resource_shape: str = "gpu.8xh200"
    allowed_nodes: list[str] = Field(default_factory=list)
    mounts: list[Mount] = [
        Mount.model_validate(
            {
                "path": "/",
                "from": "node-nfs:lepton-shared-fs",
                "mount_path": "/mnt/lepton-shared-fs",
                "mount_options": {"local_cache_size_mib": None, "read_only": None},
            }
        )
    ]
    env: dict[str, str] = Field(default_factory=dict)


class LeptonConfig(BaseModel):
    type: Literal["lepton"]
    workspace_id: str
    endpoint: LeptonEndpointConfig = LeptonEndpointConfig()
    job: LeptonJobConfig = LeptonJobConfig()
    env: dict[str, str] = Field(default_factory=dict)

    def build(self, cfg_ctx) -> DeploymentPlan:
        from domyn_swarm.backends.compute.lepton import LeptonComputeBackend
        from domyn_swarm.backends.serving.lepton import LeptonServingBackend

        serving = LeptonServingBackend(workspace=self.workspace_id)
        compute = LeptonComputeBackend(workspace=self.workspace_id)

        api_token = None
        if not self.endpoint.api_token_secret_name:
            api_token = "".join(
                secrets.choice(string.ascii_letters + string.digits) for _ in range(32)
            )

        requirement = ResourceRequirement(
            min_replicas=cfg_ctx.replicas,
            max_replicas=cfg_ctx.replicas,
            resource_shape=self.endpoint.resource_shape,
            affinity=LeptonResourceAffinity(
                allowed_dedicated_node_groups=self.endpoint.allowed_dedicated_node_groups,
                allowed_nodes_in_node_group=self.endpoint.allowed_nodes or None,
            ),
        )
        serving_container = LeptonContainer(
            image=cfg_ctx.image or self.endpoint.image,
            command=[
                "vllm",
                "serve",
                cfg_ctx.model,
                "--host",
                "0.0.0.0",
                "--port",
                str(cfg_ctx.port),
                "--tensor-parallel-size",
                str(cfg_ctx.gpus_per_replica),
                *cfg_ctx.args.split(),
            ],
        )
        serving_spec = LeptonDeploymentUserSpec(
            container=serving_container,
            resource_requirement=requirement,
            mounts=self.endpoint.mounts,
            envs=[
                EnvVar(name=k, value=v)
                for k, v in (self.env | self.endpoint.env or {}).items()
            ],
            api_tokens=[TokenVar(value=api_token)],
        ).model_dump(exclude_none=True, by_alias=True)

        job_resources = LeptonJobUserSpec(
            resource_shape=self.job.resource_shape,
            affinity=LeptonResourceAffinity(
                allowed_dedicated_node_groups=self.job.allowed_dedicated_node_groups,
                allowed_nodes_in_node_group=self.job.allowed_nodes or None,
            ),
            mounts=self.job.mounts,
            envs=[
                EnvVar(name=k, value=v)
                for k, v in (self.env | self.job.env or {}).items()
            ],
        ).model_dump(exclude_none=True, by_alias=True)

        return DeploymentPlan(
            name_hint=f"lepton-{self.workspace_id}",
            serving=serving,
            compute=compute,
            serving_spec=serving_spec,
            job_resources=job_resources,
            extras={"workspace_id": self.workspace_id, "api_token": api_token},
            platform="lepton",
        )
