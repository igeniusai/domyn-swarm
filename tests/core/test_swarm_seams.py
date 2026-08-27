# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""The public seams of DomynLLMSwarm: platform, status, rehydration, purity."""

from pathlib import Path

import pytest

from domyn_swarm.config.slurm import SlurmConfig, SlurmEndpointConfig
from domyn_swarm.config.swarm import DomynLLMSwarmConfig
from domyn_swarm.core.state.db import make_session_factory
from domyn_swarm.core.state.models import JobRecord, SwarmRecord
from domyn_swarm.core.state.state_manager import SwarmStateManager
from domyn_swarm.core.swarm import DomynLLMSwarm
from domyn_swarm.platform.protocols import ServingHandle


def _init_schema(db_path: Path) -> None:
    """Create the swarm and job tables in a temporary SQLite file."""
    session_factory = make_session_factory(db_path)
    with session_factory() as s:
        engine = s.get_bind()
        SwarmRecord.__table__.create(bind=engine, checkfirst=True)
        JobRecord.__table__.create(bind=engine, checkfirst=True)


@pytest.fixture
def db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point SwarmStateManager at a temporary SQLite DB with the schema created."""
    path = tmp_path / SwarmStateManager.DB_NAME
    monkeypatch.setattr(
        SwarmStateManager, "_get_db_path", classmethod(lambda cls: path)
    )
    _init_schema(path)
    return path


@pytest.fixture
def slurm_cfg(tmp_path: Path) -> DomynLLMSwarmConfig:
    """A minimal valid Slurm config rooted in a temporary home directory."""
    return DomynLLMSwarmConfig(
        name="fake",
        hf_home=".",
        image="image",
        model="Qwen/Qwen3-32B",
        revision=None,
        replicas=1,
        gpus_per_node=1,
        gpus_per_replica=1,
        replicas_per_node=1,
        nodes=1,
        port=1000,
        cpus_per_task=2,
        mem_per_cpu="1GB",
        wait_endpoint_s=1200,
        home_directory=tmp_path,
        backend=SlurmConfig(
            type="slurm",
            partition="partition",
            account="account",
            qos="qos",
            requires_ray=False,
            endpoint=SlurmEndpointConfig(
                cpus_per_task=1,
                nginx_image="/path/to/vllm.sif",
                mem="1GB",
                threads_per_core=1,
                wall_time="24:00:00",
                enable_proxy_buffering=True,
                nginx_timeout="60s",
            ),
        ),
    )


@pytest.fixture
def swarm(db_path: Path, slurm_cfg: DomynLLMSwarmConfig) -> DomynLLMSwarm:
    """A deployed-looking swarm wired to the temporary state DB."""
    return DomynLLMSwarm(
        name="swarm",
        cfg=slurm_cfg,
        serving_handle=ServingHandle(
            id="1234",
            url="http://lb:9003",
            meta={
                "lb_jobid": 1234,
                "jobid": 5678,
                "lb_node": "lrdn4759",
                "port": 9003,
                "name": "swarm",
            },
        ),
        endpoint="http://lb:9003",
        delete_on_exit=True,
    )


def test_platform_is_public(swarm: DomynLLMSwarm) -> None:
    """The platform is part of the public surface, not a private attribute."""
    assert swarm.platform == "slurm"
    assert not hasattr(swarm, "_platform")


def test_saved_swarm_reports_its_platform_in_list_all(swarm: DomynLLMSwarm) -> None:
    """Regression: `domyn-swarm swarm list` showed an empty backend column.

    `_platform` was a PrivateAttr, so it never reached the persisted payload
    that list_all() read it back out of.
    """
    swarm._state_mgr.save(deployment_name="swarm")

    rows = SwarmStateManager.list_all()
    row = next(r for r in rows if r["deployment_name"] == "swarm")

    assert row["platform"] == "slurm"
