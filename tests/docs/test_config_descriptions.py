# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

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
from domyn_swarm.config.settings import Settings
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
    Settings,
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
