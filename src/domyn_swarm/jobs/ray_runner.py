# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Compatibility shim for deprecated imports."""

from __future__ import annotations

import warnings

from domyn_swarm.jobs.execution.ray import _ensure_ray_initialized, run_ray_job

warnings.warn(
    "domyn_swarm.jobs.ray_runner is deprecated; use domyn_swarm.jobs.execution.ray",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["_ensure_ray_initialized", "run_ray_job"]
