# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator
import os
from pathlib import Path
import uuid

import pytest

from .cluster import queued_jobs
from .swarm import SwarmFactory

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


@pytest.fixture
def swarm_factory() -> Iterator[SwarmFactory]:
    # The images must be on the shared volume: the LB job runs nginx.sif on a compute node.
    workdir = Path("/shared/e2e") / uuid.uuid4().hex[:8]
    factory = SwarmFactory(workdir)
    try:
        yield factory
    finally:
        factory.tear_down()
