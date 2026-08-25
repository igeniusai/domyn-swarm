# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

__all__ = ["DomynLLMSwarm", "DomynLLMSwarmConfig", "SwarmJob", "__version__", "run_job_unified"]


def _resolve_version() -> str:
    """Return the installed package version without importing metadata on startup."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("domyn-swarm")
    except PackageNotFoundError:
        return "0.0.0"


def __getattr__(name: str):
    if name == "__version__":
        return _resolve_version()
    if name == "DomynLLMSwarm":
        from .core.swarm import DomynLLMSwarm

        return DomynLLMSwarm
    if name == "DomynLLMSwarmConfig":
        from .config.swarm import DomynLLMSwarmConfig

        return DomynLLMSwarmConfig
    if name == "SwarmJob":
        from .jobs.base import SwarmJob

        return SwarmJob
    raise AttributeError(name)


if TYPE_CHECKING:
    __version__: str
    from .config.swarm import DomynLLMSwarmConfig
    from .core.swarm import DomynLLMSwarm
    from .jobs.base import SwarmJob
    from .jobs.execution.dispatch import run_job_unified
