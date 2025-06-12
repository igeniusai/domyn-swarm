import pathlib

import yaml
from domyn_swarm import DomynLLMSwarmConfig, DomynLLMSwarm
import typer
from typing_extensions import Annotated

def launch_swarm(
        driver_script: Annotated[pathlib.Path, typer.Argument(file_okay=True)], 
        config: Annotated[str, typer.Option(..., "-c", "--config", help="Path to YAML config for LLMSwarmConfig")]
    ):

    # load and parse YAML
    cfg_dict = yaml.safe_load(open(config))
    cfg_dict["driver_script"] = driver_script
    # initialize dataclass with defaults, then override
    cfg = DomynLLMSwarmConfig(**cfg_dict)  # defaults
    with DomynLLMSwarm(cfg):
        pass  # all work happens inside the allocation

app = typer.Typer()
app.command()(launch_swarm)


if __name__ == "__main__":
    app()
