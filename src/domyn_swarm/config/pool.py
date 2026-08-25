# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from pydantic import BaseModel, Field
import yaml

from domyn_swarm.config.swarm import DomynLLMSwarmConfig
from domyn_swarm.core.swarm import DomynLLMSwarm
from domyn_swarm.helpers.io import to_path


class SwarmPoolElement(BaseModel):
    name: str = Field(
        description="Label identifying this swarm within the pool.",
    )
    config_path: str = Field(
        description=(
            "Path to a `DomynLLMSwarmConfig` YAML file describing this swarm. Read "
            "with `DomynLLMSwarmConfig.read` when the pool is built."
        ),
    )


class SwarmPoolConfig(BaseModel):
    pool: list[SwarmPoolElement] = Field(
        description="The swarms making up the pool, one entry per swarm.",
    )


class SwarmPool(BaseModel):
    swarms: list[DomynLLMSwarm]

    @classmethod
    def from_config(cls, path: Path | str):
        path = to_path(path)
        pool_config = SwarmPoolConfig.model_validate(yaml.safe_load(path.open()))
        return cls(
            swarms=[
                DomynLLMSwarm(cfg=DomynLLMSwarmConfig.read(element.config_path))
                for element in pool_config.pool
            ]
        )
