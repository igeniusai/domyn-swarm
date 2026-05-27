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
