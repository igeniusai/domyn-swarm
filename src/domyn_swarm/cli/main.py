from importlib import metadata
from domyn_swarm import utils
import subprocess
from typing import List, Optional
from rich import print as rprint
import typer
from typing_extensions import Annotated
import yaml

from domyn_swarm import (
    DomynLLMSwarm,
    DomynLLMSwarmConfig,
    _start_swarm,
    _load_job,
    _load_swarm_config,
    create_swarm_pool,
)
from domyn_swarm.cli.pool import pool_app
from domyn_swarm.cli.submit import submit_app

app = typer.Typer()

app.add_typer(
    submit_app, name="submit", help="Submit a workload to a Domyn-Swarm allocation."
)
app.add_typer(
    pool_app,
    name="pool",
    help="Submit a pool of swarm allocations from a YAML config.",
)


def version_callback(value: bool):
    if value:
        version = metadata.version("domyn-swarm")
        rprint(f"domyn-swarm CLI Version: {version}")
        raise typer.Exit()


@app.command("version")
def main(
    version: Annotated[
        Optional[bool], typer.Option("--version", callback=version_callback)
    ] = True,
):
    pass


@app.command("up", short_help="Launch a swarm allocation with a configuration")
def launch_up(
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
    ] = None,
):
    cfg = _load_swarm_config(config, replicas=replicas)
    _start_swarm(name, cfg, reverse_proxy=reverse_proxy)


@app.command(
    "status",
    short_help="Check the status of the swarm allocation (not yet implemented)",
)
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


@app.command("down", short_help="Shut down a swarm allocation")
def down(
    state_file: typer.FileText = typer.Argument(
        ..., exists=True, help="The swarm_*.json file printed at launch"
    ),
):
    swarm = DomynLLMSwarm.model_validate_json(state_file.read())  # validate the file
    lb = swarm.lb_jobid
    arr = swarm.jobid

    typer.echo(f"🔴  Cancelling LB  job {lb}")
    subprocess.run(["scancel", str(lb)], check=False)

    typer.echo(f"🔴  Cancelling array job {arr}")
    subprocess.run(["scancel", str(arr)], check=False)

    typer.echo("✅  Swarm shutdown request sent.")



if __name__ == "__main__":
    app()
