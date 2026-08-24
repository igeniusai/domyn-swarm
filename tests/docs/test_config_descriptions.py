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

"""Every config field that appears in the generated reference must be described.

This locks in the description backfill: a new field added without a
``description=`` fails here rather than silently rendering an empty table row.
"""

from __future__ import annotations

from pydantic import BaseModel
import pytest

from domyn_swarm.config.backend import BackendsConfig
from domyn_swarm.config.lepton import (
    LeptonConfig,
    LeptonEndpointConfig,
    LeptonJobConfig,
)
from domyn_swarm.config.pool import SwarmPoolConfig, SwarmPoolElement
from domyn_swarm.config.slurm import SlurmConfig, SlurmEndpointConfig
from domyn_swarm.config.swarm import DomynLLMSwarmConfig
from domyn_swarm.config.watchdog import WatchdogConfig, WatchdogRayConfig

DOCUMENTED_MODELS: tuple[type[BaseModel], ...] = (
    DomynLLMSwarmConfig,
    BackendsConfig,
    SlurmConfig,
    SlurmEndpointConfig,
    LeptonConfig,
    LeptonEndpointConfig,
    LeptonJobConfig,
    WatchdogConfig,
    WatchdogRayConfig,
    SwarmPoolConfig,
    SwarmPoolElement,
)


def _undescribed(model: type[BaseModel]) -> list[str]:
    return [
        name for name, field in model.model_fields.items() if not (field.description or "").strip()
    ]


@pytest.mark.parametrize("model", DOCUMENTED_MODELS, ids=lambda m: m.__name__)
def test_every_field_has_a_description(model: type[BaseModel]) -> None:
    missing = _undescribed(model)
    assert not missing, (
        f"{model.__name__} fields without Field(description=...): {missing}. "
        "Add a description; it feeds the generated configuration reference."
    )
