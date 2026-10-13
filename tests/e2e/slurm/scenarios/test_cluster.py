# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from .cluster import cancel, job_node, job_state, run, sbatch, wait_for

COMMON = ["--account=e2e", "--qos=normal", "--partition=debug", "--time=00:05:00"]


def test_compute_nodes_are_idle_with_two_gpus_each():
    rows = sorted(
        line.split() for line in run(["sinfo", "-h", "-N", "-o", "%N %T %G"]).splitlines()
    )

    # %G can append socket details, as in gpu:2(S:0).
    assert [(node, state, gres.split("(")[0]) for node, state, gres in rows] == [
        ("c1", "idle", "gpu:2"),
        ("c2", "idle", "gpu:2"),
    ]


def test_replica_and_exclusive_lb_jobs_run_on_separate_nodes():
    replicas = sbatch(
        *COMMON,
        "--exclusive=user",
        "--gres=gpu:2",
        "--mem=0",
        "--nodes=1",
        "--ntasks-per-node=2",
        "--cpus-per-task=1",
        "--wrap",
        "sleep 300",
    )
    lb = sbatch(
        *COMMON,
        "--exclusive",
        "--nodes=1",
        "--cpus-per-task=1",
        "--mem=512M",
        "--dependency",
        f"after:{replicas}",
        "--wrap",
        "sleep 300",
    )
    try:
        wait_for(lambda: job_state(replicas) == "RUNNING", 60, "replica job never ran")
        wait_for(lambda: job_state(lb) == "RUNNING", 60, "exclusive LB job never ran")
        assert job_node(replicas) != job_node(lb)

        step = run(
            [
                "srun",
                "--overlap",
                f"--jobid={replicas}",
                "-w",
                job_node(replicas),
                "-n1",
                "-N1",
                "--gres=gpu:1",
                "hostname",
            ],
            timeout=60,
        )
        assert step.strip() == job_node(replicas)
    finally:
        cancel(replicas, lb)
        wait_for(lambda: not job_state(replicas) and not job_state(lb), 60, "jobs not cancelled")
