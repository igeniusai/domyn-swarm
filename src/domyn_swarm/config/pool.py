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

from pathlib import Path

import yaml
from pydantic import BaseModel

from domyn_swarm.config.swarm import DomynLLMSwarmConfig
from domyn_swarm.core.swarm import DomynLLMSwarm
from domyn_swarm.helpers.io import to_path


class SwarmPoolElement(BaseModel):
    name: str
    config_path: str


class SwarmPoolConfig(BaseModel):
    pool: list[SwarmPoolElement]


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
