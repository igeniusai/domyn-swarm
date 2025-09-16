import logging
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from typing_extensions import Annotated

from domyn_swarm import _start_swarm
from domyn_swarm.cli.pool import pool_app
from domyn_swarm.cli.submit import submit_app
from domyn_swarm.core.state import SwarmStateManager
from domyn_swarm.helpers.logger import setup_logger
from domyn_swarm.helpers.reverse_proxy import is_endpoint_healthy
from domyn_swarm.models.swarm import _load_swarm_config
from domyn_swarm.slurm import get_job_status

app = typer.Typer(name="domyn-swarm CLI", no_args_is_help=True)

app.add_typer(
    submit_app, name="submit", help="Submit a workload to a Domyn-Swarm allocation."
)
app.add_typer(
    pool_app,
    name="pool",
    help="Submit a pool of swarm allocations from a YAML config.",
)
console = Console()
logger = setup_logger("domyn_swarm.cli", level=logging.INFO, console=console)


def version_callback(value: bool):
    if value:
        version = metadata.version("domyn-swarm")
        rprint(f"domyn-swarm CLI Version: [cyan]{version}[/cyan]")
        raise typer.Exit()


@app.command("version", short_help="Show the version of the domyn-swarm CLI")
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
    short_help="Check the status of the swarm allocation given its state file",
)
def check_status(
    jobid: int = typer.Argument(..., exists=True, help="Job ID."),  # TODO: string
    home_directory: Path = typer.Argument(
        default=Path("./.domyn_swarm"),
        help="Home directory if different from ./.domyn_swarm",
    ),
    name: Annotated[
        Optional[str],
        typer.Option(
            "--name",
            "-n",
            help="Name of the swarm allocation to check status for. If not provided, checks all allocations.",
        ),
    ] = None,
) -> None:
    """
    Check the status of the swarm allocation.

    This command will read the DB and print the status of the swarm allocation.
    If a name is provided, it will check the status of that specific allocation.
    """
    swarm = SwarmStateManager.load(jobid, home_directory)
    name = name or swarm.name
    load_balancer_jobid = swarm.lb_jobid
    array_jobid = swarm.jobid
    endpoint = swarm.endpoint
    replicas = swarm.cfg.replicas

    job_home = swarm.cfg.home_directory / "swarms" / str(array_jobid)
    how_many_vllm_endpoints = sum(1 for _ in job_home.glob("*.head"))

    vllm_endpoints = [
        f"http://{f.open().read().strip()}" for f in job_home.glob("*.head")
    ]
    vllm_status = [
        "HEALTHY" if is_endpoint_healthy(f"{ep}/v1/models") else "UNHEALTHY"
        for ep in vllm_endpoints
    ]
    vllm_endpoints_status = [
        f"{ep} ({status})" for ep, status in zip(vllm_endpoints, vllm_status)
    ]

    lb_status = (
        "HEALTHY" if is_endpoint_healthy(f"{endpoint}/v1/models") else "UNHEALTHY"
    )

    if load_balancer_jobid is None:
        raise RuntimeError("Job ID is null.")

    if array_jobid is None:
        raise RuntimeError("Job ID array is null.")

    lb_job_status = get_job_status(load_balancer_jobid)
    array_job_status = get_job_status(array_jobid)

    lb_table = Table.grid(padding=1)
    lb_table.add_row("Job ID:", str(load_balancer_jobid))
    lb_table.add_row("Status:", f"{lb_status} ({lb_job_status})")
    lb_table.add_row("Endpoint:", endpoint or "N/A")

    vllm_table = Table.grid(padding=1)
    vllm_table.add_row("Array Job ID:", str(array_jobid))
    vllm_table.add_row("Status:", array_job_status)
    vllm_table.add_row("Replicas:", f"{how_many_vllm_endpoints}/{str(replicas)}")
    vllm_table.add_row("Endpoints:", "\n".join(vllm_endpoints_status) or "N/A")

    console.print(Panel(lb_table, title="[bold cyan]Load Balancer", expand=False))
    console.print(Panel(vllm_table, title="[bold magenta]vLLM Workers", expand=False))


@app.command("down", short_help="Shut down a swarm allocation")
def down(
    jobid: int = typer.Argument(..., exists=True, help="Job ID."),
    home_directory: Path = typer.Argument(
        default=Path("./.domyn_swarm"),
        help="Home directory if different from ./.domyn_swarm",
    ),
):
    swarm = SwarmStateManager.load(jobid, home_directory)
    lb = swarm.lb_jobid
    arr = swarm.jobid

    console.print(f"🔴  Cancelling LB  job {lb}")
    subprocess.run(["scancel", str(lb)], check=False)

    console.print(f"🔴  Cancelling array job {arr}")
    subprocess.run(["scancel", str(arr)], check=False)

    typer.echo("✅  Swarm shutdown request sent.")
    swarm.delete_record()


if __name__ == "__main__":
    app()
