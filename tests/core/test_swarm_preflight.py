# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""The programmatic API must also refuse to deploy a config with bad paths."""

from pathlib import Path

import pytest

from domyn_swarm.config.slurm import SlurmConfig, SlurmEndpointConfig
from domyn_swarm.config.swarm import DomynLLMSwarmConfig
from domyn_swarm.core.swarm import DomynLLMSwarm
from domyn_swarm.exceptions import SwarmConfigPathError


@pytest.fixture
def cfg_with_missing_image(tmp_path: Path) -> DomynLLMSwarmConfig:
    """A Slurm config whose vLLM image path does not exist."""
    return DomynLLMSwarmConfig(
        name="fake",
        image="/shared/images/vllm.sif",
        model="nvidia/Nemotron-Nano-9B-v2",
        home_directory=tmp_path,
        backend=SlurmConfig(
            type="slurm",
            partition="partition",
            account="account",
            qos="qos",
            requires_ray=False,
            endpoint=SlurmEndpointConfig(nginx_image=str(tmp_path / "nginx.sif")),
        ),
    )


def test_entering_a_swarm_with_a_missing_path_raises(cfg_with_missing_image, tmp_path):
    swarm = DomynLLMSwarm(name="preflight", cfg=cfg_with_missing_image)

    with pytest.raises(SwarmConfigPathError) as excinfo:
        swarm.__enter__()

    assert [p.field for p in excinfo.value.problems] == ["image", "backend.endpoint.nginx_image"]
    assert "/shared/images/vllm.sif" in str(excinfo.value)


def test_nothing_is_created_on_disk_when_preflight_fails(cfg_with_missing_image, tmp_path):
    """Failing early means failing before the swarm directory tree is laid out."""
    before = set(tmp_path.rglob("*"))
    swarm = DomynLLMSwarm(name="preflight", cfg=cfg_with_missing_image)

    with pytest.raises(SwarmConfigPathError):
        swarm.__enter__()

    assert set(tmp_path.rglob("*")) == before


def test_preflight_can_be_disabled_on_the_swarm(cfg_with_missing_image):
    """`--skip-preflight` is only real if the library honours the opt-out too."""
    swarm = DomynLLMSwarm(name="preflight", cfg=cfg_with_missing_image, preflight=False)

    with pytest.raises(Exception) as excinfo:
        swarm.__enter__()

    assert not isinstance(excinfo.value, SwarmConfigPathError)


def test_error_message_carries_the_hint(cfg_with_missing_image):
    """The deepest-existing-parent hint is the point; it must survive the raise."""
    swarm = DomynLLMSwarm(name="preflight", cfg=cfg_with_missing_image)

    with pytest.raises(SwarmConfigPathError) as excinfo:
        swarm.__enter__()

    assert "deepest existing parent" in str(excinfo.value)
