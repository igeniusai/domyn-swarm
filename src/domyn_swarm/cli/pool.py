import pathlib
from typing_extensions import Annotated
from pydantic import BaseModel
import typer
import yaml

from domyn_swarm import DomynLLMSwarm, DomynLLMSwarmConfig, create_swarm_pool
from domyn_swarm.helpers import to_path


class SwarmPoolElement(BaseModel):
    name: str
    config_path: str


class SwarmPoolConfig(BaseModel):
    pool: list[SwarmPoolElement]


class SwarmPool(BaseModel):
    swarms: list[DomynLLMSwarm]

    @classmethod
    def from_config(cls, path: pathlib.Path | str):
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


pool_app = typer.Typer(help="Submit a pool to a Domyn-Swarm allocation.")

@pool_app.command(
    "pool",
    short_help="Deploy a pool of swarm allocations from a YAML config (not yet implemented)",
)
def deploy_pool(
    config: Annotated[
        typer.FileText,
        typer.Argument(
            file_okay=True,
            help="Path to YAML config for a pool of swarm allocations",
        ),
    ],
):
    pool_config = SwarmPoolConfig.model_validate(yaml.safe_load(config.read()))
    named_swarms = [
        DomynLLMSwarm(name=name, cfg=DomynLLMSwarmConfig.read(config_path))
        for name, config_path in pool_config.pool
    ]
    with create_swarm_pool(*named_swarms):
        pass
