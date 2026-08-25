# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from pydantic import BaseModel, Field

from domyn_swarm.config.lepton import LeptonConfig
from domyn_swarm.config.plan import DeploymentPlan
from domyn_swarm.config.slurm import SlurmConfig

BackendConfig = Annotated[
    LeptonConfig | SlurmConfig,
    Field(discriminator="type"),
]


class BackendsConfig(BaseModel):
    backends: list[BackendConfig] = Field(
        description=(
            "Backend configurations to build deployment plans for. Each entry is "
            "selected by its `type` discriminator: `slurm` or `lepton`."
        ),
    )

    def build_all(self, cfg_ctx) -> list[DeploymentPlan]:
        return [b.build(cfg_ctx) for b in self.backends]
