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
import sys
from typing import Annotated

from rich.console import Console
from rich.table import Table
import typer

from domyn_swarm.backends.serving.slurm_readiness import SwarmReplicaFailure
from domyn_swarm.core.state.watchdog import SwarmReplicaSummary, read_swarm_summary
from domyn_swarm.runtime.status import read_replica_statuses
from domyn_swarm.utils.cli import _pick_one

from ..cli.init import init_app
from ..cli.pool import pool_app
from ..cli.tui import render_status
from ..config.swarm import _load_swarm_config
from ..core.state.autoupgrade import ensure_db_up_to_date
from ..core.state.state_manager import SwarmStateManager
from ..core.swarm import DomynLLMSwarm
from ..helpers.logger import setup_logger
from ..utils.version import get_version
from .db import db_app
from .job import job_app
from .swarm import swarm_app

app = typer.Typer(name="domyn-swarm CLI", no_args_is_help=True)

app.add_typer(job_app, name="job", help="Submit a workload to a Domyn-Swarm allocation.")
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
app.add_typer(
    db_app,
    name="db",
    help="Manage the Domyn-Swarm state database.",
)

console = Console()
logger = setup_logger("domyn_swarm.cli", level=logging.INFO, console=console)


@app.callback()
def main_callback(ctx: typer.Context) -> None:
    """
    Root callback: runs before any subcommand.

    We use this to transparently auto-upgrade the local swarm.db schema.
    """
    # If the user is explicitly running `domyn-swarm db ...`,
    # let that subcommand control migrations.
    if ctx.invoked_subcommand == "db":
        return

    # Safe to call for everything else; it's idempotent and guarded.
    ensure_db_up_to_date(noisy=True)


@app.command("version", short_help="Show the version of the domyn-swarm CLI")
def version(short: bool = False):
    v = get_version()
    typer.echo(v if short else f"domyn-swarm CLI Version: {v}")
    raise typer.Exit()


@app.command("up", short_help="Launch a swarm allocation with a configuration")
def launch_up(
    config: Annotated[
        typer.FileText,
        typer.Option(..., "-c", "--config", help="Path to YAML config for LLMSwarmConfig"),
    ],
    reverse_proxy: Annotated[
        bool,
        typer.Option(
            "--reverse-proxy/--no-reverse-proxy",
            help="Enable reverse proxy for the swarm allocation",
        ),
    ] = False,
    replicas: Annotated[
        int | None,
        typer.Option(
            "--replicas",
            "-r",
            help="Number of replicas for the swarm allocation. Defaults to 1.",
        ),
    ] = None,
):
    """
    Launch a swarm allocation with the given configuration.
    The configuration must be provided as a YAML file.
    """
    cfg = _load_swarm_config(config, replicas=replicas)
    swarm_ctx = DomynLLMSwarm(cfg=cfg)

    def _safe_down(ctx: DomynLLMSwarm) -> None:
        try:
            ctx.down()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    try:
        with swarm_ctx as swarm:
            # Print ONLY the name to stdout so command substitution works if not tty:
            #   SWARM_NAME=$(domyn-swarm up -c config.yaml)
            if not sys.stdout.isatty():
                typer.echo(swarm.name)
            raise typer.Exit(code=0)
    except KeyboardInterrupt as ki:
        if typer.confirm("KeyboardInterrupt detected. Cancel the swarm allocation?"):
            _safe_down(swarm_ctx)
            typer.echo("Swarm allocation cancelled by user", err=True)
            raise typer.Abort() from ki
        typer.echo(f"Waiting for swarm {swarm_ctx.name}…", err=True)
    except SwarmReplicaFailure as srf:
        logger.error(f"Swarm replica failure detected: {srf}")
        _safe_down(swarm_ctx)
        raise typer.Exit(code=1) from srf


@app.command(
    "status",
    short_help="Check the status of the swarm allocation given its state file",
)
def check_status(
    name: Annotated[
        str,
        typer.Argument(
            ...,
            help="Name of the swarm allocation to check status for. "
            "If not provided, checks all allocations.",
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
    replica_summary: SwarmReplicaSummary | None = None
    replica_rows = []

    try:
        replica_summary = read_swarm_summary(swarm.watchdog_db_path, swarm_id=name)
    except Exception as e:
        logger.debug(f"Could not read swarm replica summary: {e}")

    try:
        replica_rows = read_replica_statuses(swarm.watchdog_db_path, swarm_id=name)
    except Exception as e:
        logger.debug(f"Could not read swarm replica rows: {e}")

    render_status(
        (name, swarm._platform, serving_status),
        replica_summary=replica_summary,
        replica_rows=replica_rows,
        console=console,
    )


def down_by_name(name: str, yes: bool) -> None:
    """
    Shut down a swarm by its name, with optional confirmation.
    This function loads a swarm by name and shuts it down. If the swarm is currently
    running and the yes parameter is False, it will prompt the user for confirmation
    before proceeding with the shutdown.
    Args:
        name (str): The name of the swarm to shut down.
        yes (bool): If True, skip confirmation prompt for running swarms.
    Returns:
        None
    Raises:
        typer.Exit: If user cancels the shutdown when prompted for confirmation.
    Example:
        >>> down_by_name("my-swarm", yes=False)
        # Will prompt for confirmation if swarm is running
        >>> down_by_name("my-swarm", yes=True)
        # Will shut down without confirmation
    """
    swarm = SwarmStateManager.load(deployment_name=name)
    serving_status = swarm.status()
    if serving_status.phase == "RUNNING" and not yes:
        confirm = typer.confirm(f"Are you sure you want to shut down the running swarm {name}?")
        if not confirm:
            typer.secho("Aborting shutdown.", fg="red")
            raise typer.Exit()
    swarm.down()
    typer.echo(f"✅ Swarm {name} shutdown request sent.")


def down_by_config(config: typer.FileText, yes: bool, all_: bool, select: bool) -> None:
    """
    Shut down swarms matching the base name in the provided config file.
    If multiple swarms match, the user can choose to shut down all, select one,
    or be prompted for confirmation.
    Args:
        config (typer.FileText): File handle to the YAML config containing the swarm name.
        yes (bool): If True, skip confirmation prompts.
        all_ (bool): If True, shut down all matching swarms.
        select (bool): If True, prompt the user to select one swarm to shut down.
    Raises:
        typer.Exit: Exits the program after shutting down swarms or if no action is taken.
    """
    logger.info(f"Loading swarm config from {config.name} to find matching swarms")
    base_name = _load_swarm_config(config).name
    matches = SwarmStateManager.list_by_base_name(base_name)

    logger.info(f"Found {len(matches)} matching swarms for base name '{base_name}'")

    if not matches:
        console.print(f"[yellow]No swarms found for base name '{base_name}'.[/]")
        raise typer.Exit(0)

    if len(matches) == 1 and not all_:
        name = matches[0]
        console.print(f"[cyan]Found 1 match:[/] {name}")
        if not yes and not typer.confirm(f"Destroy swarm '{name}'?"):
            raise typer.Abort()
        down_by_name(name=name, yes=yes)  # reuse your existing down logic
        raise typer.Exit(0)

    if all_:
        if not yes:
            table = Table(title="Swarms to destroy")
            table.add_column("deployment_name")
            for n in matches:
                table.add_row(n)
            console.print(table)
            if not typer.confirm(f"Destroy ALL {len(matches)} swarms above?"):
                raise typer.Abort()
        for n in matches:
            down_by_name(name=n, yes=True)
        raise typer.Exit(0)

    if select:
        name = _pick_one(matches, console)
        if not yes and not typer.confirm(f"Destroy swarm '{name}'?"):
            raise typer.Abort()
        down_by_name(name=name, yes=True)
        raise typer.Exit(0)

    if matches:
        console.print(
            f"[red]Multiple swarms found for base name '{base_name}'. "
            "Please specify one using --name, use --select to pick "
            "one or use --all to delete all of them.[/]"
        )
        for n in matches:
            console.print(f" - {n}")
        raise typer.Exit(1)


@app.command("down", short_help="Shut down a swarm allocation")
def down(
    name: str | None = typer.Argument(default=None, help="Swarm name."),
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Force shutdown without confirmation.",
        ),
    ] = False,
    select: bool = typer.Option(
        False, "--select", help="Pick a single swarm when multiple matches."
    ),
    all_: bool = typer.Option(False, "--all", help="Tear down all matching swarms."),
    config: typer.FileText | None = typer.Option(
        None, "-c", "--config", help="Path to YAML config (must contain 'name')."
    ),
):
    """
    Shut down a swarm allocation.

    If NAME is not provided, attempts to find the last created swarm.
    If a CONFIG is provided, uses its 'name' field to find matching swarms.
    Use --all to shut down all matching swarms, or --select to pick one interactively.
    """
    if name is None:
        logger.warning("Swarm name not provided for shutdown")
        if config is not None:
            down_by_config(config=config, yes=yes, all_=all_, select=select)

        logger.info(
            "No config provided or no matches found; attempting to shut down the last swarm."
        )
        name = SwarmStateManager.get_last_swarm_name()
        if name:
            logger.info(f"Shutting down the last swarm: [bold cyan]{name}[/]")
            down_by_name(name=name, yes=yes)
        else:
            logger.error("No swarms found to shut down.")
            raise typer.Exit(code=1)
    else:
        down_by_name(name=name, yes=yes)
    raise typer.Exit(0)


if __name__ == "__main__":
    app()
