from pydantic import BaseModel
from domyn_swarm import utils
import yaml

from domyn_swarm import DomynLLMSwarm, DomynLLMSwarmConfig
from domyn_swarm.helpers import to_path


class SwarmPoolElement(BaseModel):
    name: str
    config_path: str


class SwarmPoolConfig(BaseModel):
    pool: list[SwarmPoolElement]


class SwarmPool(BaseModel):
    swarms: list[DomynLLMSwarm]

    @classmethod
    def from_config(cls, path: utils.EnvPath | str):
        path = to_path(path)
        pool_config = SwarmPoolConfig.model_validate(yaml.safe_load(path.open()))
        return cls(
            swarms=[
                DomynLLMSwarm(
                    name=element.name, cfg=DomynLLMSwarmConfig.read(element.config_path)
                )
                for element in pool_config.pool
            ]
        )
