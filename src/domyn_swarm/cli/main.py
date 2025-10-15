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

import logging
from typing import Optional

import typer
from rich.console import Console
from typing_extensions import Annotated

from ..cli.init import init_app
from ..cli.pool import pool_app
from ..cli.tui import render_status
from ..config.swarm import _load_swarm_config
from ..core.state import SwarmStateManager
from ..core.swarm import DomynLLMSwarm
from ..helpers.logger import setup_logger
from ..utils.version import get_version
from .job import job_app
from .swarm import swarm_app

app = typer.Typer(name="domyn-swarm CLI", no_args_is_help=True)

app.add_typer(
    job_app, name="job", help="Submit a workload to a Domyn-Swarm allocation."
)
app.add_typer(
    pool_app,
    name="pool",
    help="Submit a pool of swarm allocations from a YAML config.",
)
app.add_typer(
    init_app,
    name="init",
    help="Initialize a new Domyn-Swarm configuration.",
)
app.add_typer(swarm_app, name="swarm")

console = Console()
logger = setup_logger("domyn_swarm.cli", level=logging.INFO, console=console)


@app.command("version", short_help="Show the version of the domyn-swarm CLI")
def version(short: bool = False):
    v = get_version()
    print(v if short else f"domyn-swarm CLI Version: {v}")
    raise typer.Exit()


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
    swarm_ctx = DomynLLMSwarm(cfg=cfg)
    try:
        with swarm_ctx as _:
            ...
    except KeyboardInterrupt:
        abort = typer.confirm(
            "KeyboardInterrupt detected. Do you want to cancel the swarm allocation?"
        )
        if abort:
            try:
                swarm_ctx.down()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
                pass
            typer.echo("Swarm allocation cancelled by user")
            raise typer.Abort()
        else:
            typer.echo(f"Waiting for swarm {swarm_ctx.name}…")


@app.command(
    "status",
    short_help="Check the status of the swarm allocation given its state file",
)
def check_status(
    name: Annotated[
        str,
        typer.Argument(
            ...,
            help="Name of the swarm allocation to check status for. If not provided, checks all allocations.",
        ),
    ],
) -> None:
    """
    Check the status of the swarm allocation.

    This command will read the DB and print the status of the swarm allocation.
    If a name is provided, it will check the status of that specific allocation.
    """
    swarm = SwarmStateManager.load(deployment_name=name)
    if swarm.serving_handle is None:
        raise ValueError("Swarm does not have a serving handle.")

    serving_status = swarm.status()
    render_status((name, swarm._platform, serving_status), console=console)


@app.command("down", short_help="Shut down a swarm allocation")
def down(
    name: str = typer.Argument(..., exists=True, help="Swarm name."),
):
    swarm = SwarmStateManager.load(deployment_name=name)
    swarm.down()
    typer.echo("✅ Swarm shutdown request sent.")
    swarm.delete_record(deployment_name=name)


if __name__ == "__main__":
    app()
