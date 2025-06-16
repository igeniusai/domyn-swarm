import pathlib
from typing import Optional

import yaml
from domyn_swarm import DomynLLMSwarmConfig, DomynLLMSwarm
import typer
from typing_extensions import Annotated

from domyn_swarm.helpers import launch_reverse_proxy

app = typer.Typer()

@app.command("up", short_help="Launch a swarm allocation with a driver script and configuration")
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
    name: Annotated[
        Optional[str],
        typer.Option(
            "--name",
            "-n",
            help="Name of the swarm allocation. If not provided, a random name will be generated.",
        ),
    ],
    replicas: Annotated[
        Optional[int],
        typer.Option(
            "--replicas",
            "-r",
            help="Number of replicas for the swarm allocation. Defaults to 1.",
            default=1,
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

@app.command("status", short_help="Check the status of the swarm allocation")
def check_status(
    name: Annotated[
        Optional[str],
        typer.Option(
            "--name",
            "-n",
            help="Name of the swarm allocation to check status for. If not provided, checks all allocations.",
        ),
    ],
):
    swarm = DomynLLMSwarm()
    if name:
        status = swarm.get_allocation_status(name)
        typer.echo(f"Status of allocation '{name}': {status}")
    else:
        allocations = swarm.list_allocations()
        typer.echo("Current allocations:")
        for alloc in allocations:
            typer.echo(f"- {alloc['name']}: {alloc['status']}")

@app.command("pool", short_help="Deploy a pool of swarm allocations")
def deploy_pool(
    config: Annotated[
        typer.FileText,
        typer.Argument(
            file_okay=True,
            help="Path to YAML config for a pool of swarm allocations",
        ),
    ]
):
    pass

@app.command("down", short_help="Terminate a swarm allocation")
def terminate_swarm(
    name: Annotated[
        Optional[str],
        typer.Option(
            "--name",
            "-n",
            help="Name of the swarm allocation to terminate. If not provided, terminates all allocations.",
        ),
    ],
):
    swarm = DomynLLMSwarm()
    if name:
        swarm.terminate_allocation(name)
        typer.echo(f"Terminated allocation '{name}'")
    else:
        swarm.terminate_all_allocations()
        typer.echo("Terminated all allocations")

if __name__ == "__main__":
    app()
