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

from typing import Annotated

import typer

pool_app = typer.Typer(help="Submit a pool to a Domyn-Swarm allocation.")


class _LazySwarmPoolConfig:
    """Proxy for pool config validation."""

    def model_validate(self, *args, **kwargs):
        """Validate pool configuration data."""
        from domyn_swarm.config.pool import SwarmPoolConfig

        return SwarmPoolConfig.model_validate(*args, **kwargs)


class _LazyDomynLLMSwarmConfig:
    """Proxy for reading swarm config files."""

    def read(self, *args, **kwargs):
        """Read a swarm config file."""
        from domyn_swarm.config.swarm import DomynLLMSwarmConfig

        return DomynLLMSwarmConfig.read(*args, **kwargs)


class _LazyDomynLLMSwarm:
    """Proxy that imports the swarm implementation only during pool deployment."""

    def __call__(self, *args, **kwargs):
        from domyn_swarm.core.swarm import DomynLLMSwarm

        return DomynLLMSwarm(*args, **kwargs)


SwarmPoolConfig = _LazySwarmPoolConfig()
DomynLLMSwarmConfig = _LazyDomynLLMSwarmConfig()
DomynLLMSwarm = _LazyDomynLLMSwarm()


def create_swarm_pool(*args, **kwargs):
    """Create a swarm pool lazily."""
    from domyn_swarm.core.swarm_pool import create_swarm_pool as create_pool

    return create_pool(*args, **kwargs)


@pool_app.command(
    "pool",
    short_help="Deploy a pool of swarm allocations from a YAML config (not yet implemented)",
)
def deploy_pool(
    config: Annotated[
        typer.FileText,
        typer.Argument(
            file_okay=True,
            help="Path to YAML config for a pool of swarm allocations",
        ),
    ],
):
    import yaml

    pool_config = SwarmPoolConfig.model_validate(yaml.safe_load(config.read()))
    named_swarms = [
        DomynLLMSwarm(
            cfg=DomynLLMSwarmConfig.read(pool_element.config_path),
        )
        for pool_element in pool_config.pool
    ]
    with create_swarm_pool(*named_swarms):
        pass
