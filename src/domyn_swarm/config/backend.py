# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from pydantic import BaseModel, Field

from domyn_swarm.config.lepton import LeptonConfig
from domyn_swarm.config.slurm import SlurmConfig

BackendConfig = Annotated[
    LeptonConfig | SlurmConfig,
    Field(discriminator="type"),
]


class BackendsConfig(BaseModel):
    """Schema for a list of backend configurations.

    Not consumed by the runtime today -- a swarm config carries a single
    `backend`. It is kept as the declared shape for multi-backend configs and is
    covered by the config-description test.
    """

    backends: list[BackendConfig] = Field(
        description=(
            "Backend configurations to build deployment plans for. Each entry is "
            "selected by its `type` discriminator: `slurm` or `lepton`."
        ),
    )
