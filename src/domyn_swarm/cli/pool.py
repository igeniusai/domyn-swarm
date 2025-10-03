# Copyright 2025 Domyn
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

import typer
import yaml
from typing_extensions import Annotated

from domyn_swarm import DomynLLMSwarm, DomynLLMSwarmConfig
from domyn_swarm.config.pool import SwarmPoolConfig
from domyn_swarm.core.swarm_pool import create_swarm_pool

pool_app = typer.Typer(help="Submit a pool to a Domyn-Swarm allocation.")


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
    pool_config = SwarmPoolConfig.model_validate(yaml.safe_load(config.read()))
    named_swarms = [
        DomynLLMSwarm(
            cfg=DomynLLMSwarmConfig.read(pool_element.config_path),
        )
        for pool_element in pool_config.pool
    ]
    with create_swarm_pool(*named_swarms):
        pass
