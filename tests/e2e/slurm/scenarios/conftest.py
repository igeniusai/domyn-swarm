# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

import pytest

from .cluster import queued_jobs

SCENARIOS = Path(__file__).parent


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    enabled = os.environ.get("DOMYN_SWARM_E2E") == "slurm"
    timeout = int(os.environ.get("DOMYN_SWARM_E2E_TIMEOUT_S", "600"))
    skip = pytest.mark.skip(reason="needs the Slurm e2e cluster: run tests/e2e/slurm/run.sh")
    for item in items:
        if SCENARIOS not in item.path.parents:
            continue
        item.add_marker(pytest.mark.e2e)
        item.add_marker(pytest.mark.timeout(timeout))
        if not enabled:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def empty_queue() -> None:
    """Fail fast when an earlier scenario left jobs behind."""
    leftover = queued_jobs()
    assert leftover == [], f"jobs left in the queue by an earlier scenario: {leftover}"
