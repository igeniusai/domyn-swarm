from leptonai.api.v1.types.deployment import Mount
from pydantic import BaseModel


class LeptonEndpointConfig(BaseModel):
    node_group: str | None = None
    resource_shape: str = "gpu.8xh200"
    allowed_nodes: list[str] | None = None
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


class LeptonJobConfig(BaseModel):
    node_group: str | None = None
    image: str = "igenius/domyn-swarm:latest"
    resource_shape: str = "gpu.8xh200"
    allowed_nodes: list[str] | None = None
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


class LeptonConfig(BaseModel):
    workspace_id: str
    endpoint: LeptonEndpointConfig = LeptonEndpointConfig()
    job: LeptonJobConfig = LeptonJobConfig()
