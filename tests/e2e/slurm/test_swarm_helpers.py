# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess

import pytest

from domyn_swarm.config.swarm import DomynLLMSwarmConfig

from .scenarios import swarm as swarm_module
from .scenarios.swarm import Swarm, SwarmFactory, parse_swarm_name, swarm_config, write_images


def test_parse_swarm_name_ignores_log_lines():
    stdout = (
        "INFO  Submitting replicas job e2e-ab12cd\n"
        "INFO  Endpoint ready at http://c2:9000\n"
        "e2e-ab12cd-01m4tfyhj0\n"
    )

    assert parse_swarm_name(stdout, "e2e-ab12cd") == "e2e-ab12cd-01m4tfyhj0"


def test_parse_swarm_name_without_a_name_fails():
    with pytest.raises(AssertionError, match="no swarm name"):
        parse_swarm_name("INFO  nothing here\n", "e2e-ab12cd")


def test_swarm_config_validates_to_two_colocated_replicas(tmp_path):
    cfg = DomynLLMSwarmConfig.model_validate(
        swarm_config(name="e2e-ab12cd", images=write_images(tmp_path))
    )

    assert cfg.replicas == 2
    assert cfg.replicas_per_node == 2
    assert cfg.nodes == 1
    assert cfg.backend.requires_ray is False


def test_tear_down_cancels_jobs_when_down_hangs(monkeypatch, tmp_path):
    jobs = [("7", "e2e-ab12cd-01m4tfyhj0"), ("8", "e2e-ab12cd-01m4tfyhj0-lb")]
    cancelled: list[str] = []

    def hanging_cli(*args: str, timeout: float = 120) -> None:
        raise subprocess.TimeoutExpired(cmd=["domyn-swarm", *args], timeout=timeout)

    def cancel(*job_ids: str) -> None:
        cancelled.extend(job_ids)
        jobs.clear()

    monkeypatch.setattr(swarm_module, "cli", hanging_cli)
    monkeypatch.setattr(swarm_module, "queued_jobs", lambda: list(jobs))
    monkeypatch.setattr(swarm_module, "cancel", cancel)
    factory = SwarmFactory(tmp_path)
    factory.config_names.append("e2e-ab12cd")
    factory.started.append(
        Swarm(name="e2e-ab12cd-01m4tfyhj0", config_name="e2e-ab12cd", endpoint="http://c2:9000")
    )

    factory.tear_down()

    assert cancelled == ["7", "8"]
