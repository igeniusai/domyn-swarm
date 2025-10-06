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

import secrets
import string
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, field_validator

from domyn_swarm.config.defaults import default_for
from domyn_swarm.config.plan import DeploymentPlan
from domyn_swarm.config.settings import get_settings
from domyn_swarm.utils.imports import _require_lepton

if TYPE_CHECKING:
    from leptonai.api.v1.types.deployment import Mount as _Mount

    MountLike = _Mount  # editors see the real type
else:
    MountLike = Any  # runtime stays lightweight (no lepton import)


def _default_mounts() -> list[dict[str, Any]] | list[MountLike]:
    """Lazy default: return a ready Mount if SDK is present, else a dict."""
    spec = {
        "path": "/",
        "from": "node-nfs:lepton-shared-fs",
        "mount_path": "/mnt/lepton-shared-fs",
        "mount_options": {"local_cache_size_mib": None, "read_only": None},
    }
    try:
        from leptonai.api.v1.types.deployment import Mount  # type: ignore

        return [Mount.model_validate(spec, by_alias=True)]
    except Exception:
        return [spec]


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
    mounts: list[MountLike] = Field(default_factory=_default_mounts)  # type: ignore
    env: dict[str, str] = Field(default_factory=dict)
    api_token_secret_name: str | None = None
    image_pull_secrets: list[str] | None = Field(default=None)

    @field_validator("mounts", mode="before")
    @classmethod
    def _coerce_mounts(cls, v: Any) -> Any:
        if v is None:
            return []
        try:
            from leptonai.api.v1.types.deployment import Mount
        except Exception:
            return v
        # Normalize to Mount instances
        out = []
        for item in v:
            if isinstance(item, Mount):
                out.append(item)
            else:
                out.append(Mount.model_validate(item, by_name=True))
        return out


class LeptonJobConfig(BaseModel):
    allowed_dedicated_node_groups: list[str] | None = None
    image: str = "igeniusai/domyn-swarm:latest"
    resource_shape: str = "gpu.8xh200"
    allowed_nodes: list[str] = Field(default_factory=list)
    mounts: list[MountLike] = Field(default_factory=_default_mounts)  # type: ignore
    env: dict[str, str] = Field(default_factory=dict)
    image_pull_secrets: list[str] | None = Field(default=None)

    @field_validator("mounts", mode="before")
    @classmethod
    def _coerce_mounts(cls, v: Any) -> Any:
        if v is None:
            return []
        try:
            from leptonai.api.v1.types.deployment import Mount
        except Exception:
            return v
        # Normalize to Mount instances
        out = []
        for item in v:
            if isinstance(item, Mount):
                out.append(item)
            else:
                out.append(Mount.model_validate(item, by_name=True))
        return out


class LeptonConfig(BaseModel):
    type: Literal["lepton"]
    workspace_id: str = Field(
        default_factory=default_for(
            "lepton.workspace_id", get_settings().lepton_workspace_id
        ),
        description="Lepton workspace ID",
    )
    endpoint: LeptonEndpointConfig = LeptonEndpointConfig()
    job: LeptonJobConfig = LeptonJobConfig()
    env: dict[str, str] = Field(default_factory=dict)

    def build(self, cfg_ctx) -> DeploymentPlan:
        from domyn_swarm.backends.compute.lepton import LeptonComputeBackend
        from domyn_swarm.backends.serving.lepton import LeptonServingBackend

        _require_lepton()
        from leptonai.api.v1.types.affinity import LeptonResourceAffinity
        from leptonai.api.v1.types.deployment import (
            EnvVar,
            LeptonContainer,
            LeptonDeploymentUserSpec,
            ResourceRequirement,
            TokenVar,
        )
        from leptonai.api.v1.types.job import LeptonJobUserSpec

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
            image_pull_secrets=self.endpoint.image_pull_secrets,
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
            image_pull_secrets=self.job.image_pull_secrets,
        ).model_dump(by_alias=True)

        return DeploymentPlan(
            name_hint=f"lepton-{self.workspace_id}",
            serving=serving,
            compute=compute,
            serving_spec=serving_spec,
            job_resources=job_resources,
            extras={"workspace_id": self.workspace_id, "api_token": api_token},
            platform="lepton",
        )
