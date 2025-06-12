import pathlib
from typing import Optional

import yaml
from domyn_swarm import DomynLLMSwarmConfig, DomynLLMSwarm
import typer
from typing_extensions import Annotated

from domyn_swarm.helpers import launch_reverse_proxy


def launch_swarm(
    driver_script: Annotated[
        pathlib.Path,
        typer.Argument(
            file_okay=True,
            help="Path to the driver script that will be executed inside the swarm allocation",
        ),
    ],
    config: Annotated[
        typer.FileText,
        typer.Option(
            ..., "-c", "--config", help="Path to YAML config for LLMSwarmConfig"
        ),
    ],
    reverse_proxy: Annotated[
       Optional[bool],
        typer.Option(
            "--reverse-proxy/--no-reverse-proxy",
            help="Enable reverse proxy for the swarm allocation",
        ),
    ],
):
    # load and parse YAML
    cfg_dict = yaml.safe_load(config)
    cfg_dict["driver_script"] = driver_script
    # initialize dataclass with defaults, then override
    cfg = DomynLLMSwarmConfig(**cfg_dict)  # defaults
    with DomynLLMSwarm(cfg) as swarm:
        if reverse_proxy:
            # this will start the reverse proxy in the background
            launch_reverse_proxy(
                cfg.nginx_template_path,
                cfg.nginx_image,
                swarm.head_node,
                cfg.vllm_port,
                cfg.ray_dashboard_port,
            )


app = typer.Typer()
app.command()(launch_swarm)


if __name__ == "__main__":
    app()
