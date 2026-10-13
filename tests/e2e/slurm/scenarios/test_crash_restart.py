# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
import urllib.error

import pytest

from .cluster import run
from .swarm import chat, model, slurm_job_id, status, tier, watchdog_rows

pytestmark = pytest.mark.skipif(
    tier() != "fast", reason="the vLLM tier runs only the lifecycle scenario"
)

RESTART_TIMEOUT_S = 120


def test_killed_replica_restarts_while_endpoint_keeps_serving(swarm_factory):
    swarm = swarm_factory.up()
    before = watchdog_rows(swarm.name)
    victim, survivor = before[0], before[1]
    replica_job = slurm_job_id(status(swarm.name), "swarm")

    run(
        [
            "srun",
            "--overlap",
            f"--jobid={replica_job}",
            "-w",
            victim["node"],
            "-n1",
            "-N1",
            "kill",
            "-9",
            str(victim["pid"]),
        ],
        timeout=60,
    )

    failures: list[str] = []
    restarted = None
    deadline = time.monotonic() + RESTART_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            chat(swarm.endpoint, model())
        except (urllib.error.URLError, OSError) as error:
            failures.append(repr(error))
        row = watchdog_rows(swarm.name)[0]
        if row["pid"] not in (None, victim["pid"]) and (row["state"], row["http_ready"]) == (
            "running",
            1,
        ):
            restarted = row
            break
        time.sleep(0.5)

    assert restarted is not None, f"replica 0 did not restart: {watchdog_rows(swarm.name)[0]}"
    assert failures == [], f"requests failed during the restart: {failures}"
    assert watchdog_rows(swarm.name)[1]["pid"] == survivor["pid"]
