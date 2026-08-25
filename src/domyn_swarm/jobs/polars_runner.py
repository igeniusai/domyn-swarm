# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Compatibility shim for deprecated imports."""

from __future__ import annotations

import warnings

from domyn_swarm.jobs.execution.polars import PolarsJobRunner, PolarsRunnerConfig, run_polars_job

warnings.warn(
    "domyn_swarm.jobs.polars_runner is deprecated; use domyn_swarm.jobs.execution.polars",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["PolarsJobRunner", "PolarsRunnerConfig", "run_polars_job"]
