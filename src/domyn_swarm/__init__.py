from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

try:
    __version__ = version("domyn_swarm")
except PackageNotFoundError:  # during dev
    __version__ = "0.0.0"

__all__ = ["DomynLLMSwarm", "DomynLLMSwarmConfig", "SwarmJob", "run_job_unified"]


def __getattr__(name: str):
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
    from .config.swarm import DomynLLMSwarmConfig
    from .core.swarm import DomynLLMSwarm
    from .jobs.base import SwarmJob
    from .jobs.compat import run_job_unified
