# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Shared CLI rendering for errors that deserve a report rather than a traceback."""

from __future__ import annotations

from collections.abc import Iterator
import contextlib

import typer

#: Exit status used for a config that cannot be deployed as written.
CONFIG_ERROR_EXIT_CODE = 2


@contextlib.contextmanager
def exit_on_config_path_error(*, source: str | None = None) -> Iterator[None]:
    """Turn a :class:`SwarmConfigPathError` into a report and a clean exit.

    Every command that builds a swarm from a YAML file inherits the path check
    in ``DomynLLMSwarm.__enter__``, and would otherwise show a traceback.

    Args:
        source: Name of the config file being deployed, when known.

    Yields:
        None.

    Raises:
        typer.Exit: With :data:`CONFIG_ERROR_EXIT_CODE` when a path is unusable.
    """
    from domyn_swarm.config.preflight import format_path_problems
    from domyn_swarm.exceptions import SwarmConfigPathError

    try:
        yield
    except SwarmConfigPathError as exc:
        typer.echo(format_path_problems(exc.problems, source=source), err=True)
        raise typer.Exit(code=CONFIG_ERROR_EXIT_CODE) from exc
