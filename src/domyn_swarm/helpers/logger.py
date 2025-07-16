import logging

from rich.console import Console
from rich.logging import RichHandler


def setup_logger(
    name: str = "app", level=logging.INFO, console: Console = None
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Avoid duplicate logs

    if logger.handlers:
        return logger

    # Console setup
    stdout_console = console or Console()  # Use full terminal width
    stderr_console = Console(stderr=True)

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
    stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)

    # Warnings and above → stderr
    stderr_handler = RichHandler(
        level=logging.WARNING,
        console=stderr_console,
        rich_tracebacks=True,
        markup=True,
        show_time=True,
        show_level=True,
        show_path=False,
    )

    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)

    return logger
