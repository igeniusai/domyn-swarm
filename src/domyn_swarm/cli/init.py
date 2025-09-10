import logging

import typer

from domyn_swarm.helpers.logger import setup_logger

logger = setup_logger("domyn_swarm.cli", level=logging.INFO)

init_app = typer.Typer(help="Initialize a new Domyn-Swarm configuration.")


@init_app.command(
    "defaults", help="Create a defaults.yaml configuration file to be used later."
)
def create_defaults(
    output: str = typer.Option(
        "defaults.yaml",
        "-o",
        "--output",
        help="Path to save the defaults YAML configuration file.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite existing file if it exists.",
    ),
):
    """
    Create a default Domyn-Swarm configuration YAML file.
    This file can be edited and used later with the `--config` option in other commands.
    """
