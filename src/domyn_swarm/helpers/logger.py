# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import logging
import sys

from rich.console import Console
from rich.logging import RichHandler


def setup_logger(
    name: str = "app",
    level=logging.INFO,
    console: Console | None = None,
    to_stderr: bool = False,
) -> logging.Logger:
    machine_mode = not sys.stdout.isatty()
    to_stderr = to_stderr or machine_mode

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Avoid duplicate logs

    if logger.handlers:
        return logger

    # Console setup
    stdout_console = console or Console()  # Use full terminal width
    stderr_console = Console(stderr=True)

    if to_stderr:
        stderr_handler = RichHandler(
            level=level,
            console=stderr_console,
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_level=True,
            show_path=False,
        )
        logger.addHandler(stderr_handler)
        return logger

    # Info and below → stdout
    stdout_handler = RichHandler(
        level=logging.DEBUG,
        console=stdout_console,
        rich_tracebacks=False,
        markup=True,
        show_time=True,
        show_level=True,
        show_path=False,
    )
    stderr_handler = RichHandler(
        level=logging.WARNING,
        console=stderr_console,
        rich_tracebacks=True,
        markup=True,
        show_time=True,
        show_level=True,
        show_path=False,
    )

    stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)

    # Warnings and above → stderr
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)

    return logger
