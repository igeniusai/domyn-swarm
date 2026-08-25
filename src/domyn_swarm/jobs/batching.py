# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Compatibility shim for deprecated imports."""

from __future__ import annotations

import warnings

from domyn_swarm.jobs.api.batching import BatchExecutor

warnings.warn(
    "domyn_swarm.jobs.batching is deprecated; use domyn_swarm.jobs.api.batching",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["BatchExecutor"]
