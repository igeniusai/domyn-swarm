from leptonai.api.v1.types.deployment import Mount, MountOptions
from pydantic import BaseModel


class LeptonEndpointConfig(BaseModel):
    node_group: str = "nv-domyn-nebius-h200-01-lznuhuob"
    resource_shape: str = "gpu.8xh200"
    allowed_nodes: list[str] | None = None
    mounts: list[Mount] = [
        Mount(
            from_="node-nfs:lepton-shared-fs",  # type: ignore
            path="/",
            mount_path="/mnt/lepton-shared-fs",
            mount_options=MountOptions(),
        )
    ]


class LeptonJobConfig(BaseModel):
    node_group: str | None = None
    image: str | None = None
    resource_shape: str = "gpu.8xh200"
    allowed_nodes: list[str] | None = None
    mounts: list[Mount] = [
        Mount(
            from_="node-nfs:lepton-shared-fs",  # type: ignore
            path="/",
            mount_path="/mnt/lepton-shared-fs",
            mount_options=MountOptions(),
        )
    ]


class LeptonConfig(BaseModel):
    workspace_id: str
    endpoint: LeptonEndpointConfig = LeptonEndpointConfig()
    job: LeptonJobConfig = LeptonJobConfig()
