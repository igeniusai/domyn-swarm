from typing import Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from domyn_swarm.config.lepton import LeptonConfig
from domyn_swarm.config.plan import DeploymentPlan
from domyn_swarm.config.slurm import SlurmConfig

BackendConfig = Annotated[
    Union[LeptonConfig, SlurmConfig],
    Field(discriminator="type"),
]


class BackendsConfig(BaseModel):
    backends: list[BackendConfig]

    def build_all(self, cfg_ctx) -> list[DeploymentPlan]:
        return [b.build(cfg_ctx) for b in self.backends]
