# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from .cluster import queued_jobs, wait_for
from .swarm import chat, cli, model, slurm_job_id, status


def test_up_status_chat_down(swarm_factory):
    swarm = swarm_factory.up()

    payload = status(swarm.name)
    assert payload["ready"] is True
    replicas = sorted(payload["replicas"]["items"], key=lambda r: r["replica_id"])
    assert [(r["replica_id"], r["state"], r["http_ready"]) for r in replicas] == [
        (0, "running", True),
        (1, "running", True),
    ]
    assert replicas[0]["node"] == replicas[1]["node"]

    reply = chat(swarm.endpoint, model())
    assert reply["model"] == model()
    assert reply["choices"][0]["message"]["role"] == "assistant"

    job_ids = {slurm_job_id(payload, "swarm"), slurm_job_id(payload, "load_balancer")}
    down = cli("down", swarm.name, "-y")
    assert down.returncode == 0, down.stderr
    wait_for(
        lambda: not job_ids & {job_id for job_id, _ in queued_jobs()},
        120,
        f"jobs {job_ids} still queued after down",
    )
