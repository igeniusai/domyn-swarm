# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Slurm queries for the end-to-end scenarios, which run on the controller."""

from __future__ import annotations

from collections.abc import Callable
import subprocess
import time


def run(args: list[str], timeout: float = 30) -> str:
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    assert result.returncode == 0, f"{args} failed ({result.returncode}):\n{result.stderr}"
    return result.stdout


def wait_for(
    predicate: Callable[[], bool], timeout: float, message: str, interval: float = 1.0
) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out after {timeout}s: {message}")
        time.sleep(interval)


def job_state(job_id: str) -> str:
    return run(["squeue", "-h", "-j", job_id, "-o", "%T"]).strip()


def job_node(job_id: str) -> str:
    return run(["squeue", "-h", "-j", job_id, "-o", "%N"]).strip()


def queued_jobs() -> list[tuple[str, str]]:
    lines = run(["squeue", "-h", "-o", "%A %j"]).splitlines()
    return [tuple(line.split(maxsplit=1)) for line in lines if line.strip()]  # type: ignore[misc]


def sbatch(*args: str) -> str:
    return run(["sbatch", "--parsable", *args]).strip().split(";")[0]


def cancel(*job_ids: str) -> None:
    if job_ids:
        subprocess.run(["scancel", *job_ids], capture_output=True, check=False)
