# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Compatibility shim for deprecated imports."""

from __future__ import annotations

import warnings

from domyn_swarm.checkpoint.store import ParquetShardStore
from domyn_swarm.jobs.api.base import OutputJoinMode
from domyn_swarm.jobs.api.runner import (
    JobRunner,
    RunnerConfig,
    normalize_batch_outputs,
    run_sharded,
)

warnings.warn(
    "domyn_swarm.jobs.runner is deprecated; use domyn_swarm.jobs.api.runner",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "JobRunner",
    "OutputJoinMode",
    "ParquetShardStore",
    "RunnerConfig",
    "normalize_batch_outputs",
    "run_sharded",
]
