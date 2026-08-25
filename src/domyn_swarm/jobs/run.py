# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Compatibility shim for deprecated imports."""

from __future__ import annotations

import warnings

from domyn_swarm.jobs.cli.run import (
    _amain,
    _load_cls,
    _write_result,
    build_job_from_args,
    main,
    parse_args,
)
from domyn_swarm.jobs.execution.dispatch import run_job_unified

warnings.warn(
    "domyn_swarm.jobs.run is deprecated; use domyn_swarm.jobs.cli.run",
    DeprecationWarning,
    stacklevel=2,
)


__all__ = [
    "_amain",
    "_load_cls",
    "_write_result",
    "build_job_from_args",
    "main",
    "parse_args",
    "run_job_unified",
]


if __name__ == "__main__":
    main()
