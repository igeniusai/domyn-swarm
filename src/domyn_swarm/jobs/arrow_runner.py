# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Compatibility shim for deprecated imports."""

from __future__ import annotations

import warnings

from domyn_swarm.jobs.execution.arrow import ArrowJobRunner, ArrowRunnerConfig, run_arrow_job

warnings.warn(
    "domyn_swarm.jobs.arrow_runner is deprecated; use domyn_swarm.jobs.execution.arrow",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ArrowJobRunner", "ArrowRunnerConfig", "run_arrow_job"]
