import json
import pathlib
import subprocess
from typing import Optional

import yaml
from domyn_swarm import DomynLLMSwarmConfig, DomynLLMSwarm
import typer
from typing_extensions import Annotated

from domyn_swarm.helpers import launch_reverse_proxy

app = typer.Typer()


@app.command(
    "run", short_help="Launch a swarm allocation with a driver script and configuration"
)
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
        bool,
        typer.Option(
            "--reverse-proxy/--no-reverse-proxy",
            help="Enable reverse proxy for the swarm allocation",
        ),
    ] = False,
    name: Annotated[
        Optional[str],
        typer.Option(
            "--name",
            "-n",
            help="Name of the swarm allocation. If not provided, a random name will be generated.",
        ),
    ] = None,
    replicas: Annotated[
        Optional[int],
        typer.Option(
            "--replicas",
            "-r",
            help="Number of replicas for the swarm allocation. Defaults to 1.",
        ),
    ] = 1,
):
    # load and parse YAML
    cfg_dict = yaml.safe_load(config)
    cfg_dict["driver_script"] = driver_script
    # initialize dataclass with defaults, then override
    cfg = DomynLLMSwarmConfig(**cfg_dict)  # defaults
    if replicas:
        cfg.replicas = replicas
    with DomynLLMSwarm(name, cfg) as swarm:
        swarm.submit_script(cfg.driver_script)
        if reverse_proxy:
            # this will start the reverse proxy in the background
            launch_reverse_proxy(
                cfg.nginx_template_path,
                cfg.nginx_image,
                swarm.lb_node,
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
    pass


@app.command("pool", short_help="Deploy a pool of swarm allocations")
def deploy_pool(
    config: Annotated[
        typer.FileText,
        typer.Argument(
            file_okay=True,
            help="Path to YAML config for a pool of swarm allocations",
        ),
    ],
):
    pass


@app.command("down")
def down(
    state_file: pathlib.Path = typer.Argument(
        ..., exists=True, help="The swarm_*.json file printed at launch"
    ),
):
    ids = json.loads(state_file.read_text())
    lb = ids["lb_job_id"]
    arr = ids["array_job_id"]

    typer.echo(f"🔴  Cancelling LB  job {lb}")
    subprocess.run(["scancel", str(lb)], check=False)

    typer.echo(f"🔴  Cancelling array job {arr}")
    subprocess.run(["scancel", str(arr)], check=False)

    typer.echo("✅  Swarm shutdown request sent.")


if __name__ == "__main__":
    app()
