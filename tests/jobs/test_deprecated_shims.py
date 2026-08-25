# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""The deprecated `domyn_swarm.jobs.*` modules must re-export, never reimplement.

`jobs/runner.py` once carried its own copy of `run_sharded`, so the deprecated
path could drift from the supported one. These tests pin every shim to identity
with its target so a duplicate cannot be reintroduced silently.
"""

from __future__ import annotations

import importlib
import warnings

import pytest

SHIMS = {
    "domyn_swarm.jobs.base": "domyn_swarm.jobs.api.base",
    "domyn_swarm.jobs.batching": "domyn_swarm.jobs.api.batching",
    "domyn_swarm.jobs.chat_completion": "domyn_swarm.jobs.api.chat_completion",
    "domyn_swarm.jobs.runner": "domyn_swarm.jobs.api.runner",
    "domyn_swarm.jobs.arrow_runner": "domyn_swarm.jobs.execution.arrow",
    "domyn_swarm.jobs.polars_runner": "domyn_swarm.jobs.execution.polars",
    "domyn_swarm.jobs.ray_runner": "domyn_swarm.jobs.execution.ray",
}


def _import(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return importlib.import_module(name)


@pytest.mark.parametrize(("shim_name", "target_name"), sorted(SHIMS.items()))
def test_shim_reexports_rather_than_reimplements(shim_name, target_name):
    shim = _import(shim_name)
    target = _import(target_name)

    reimplemented = [
        name
        for name in shim.__all__
        if hasattr(target, name) and getattr(shim, name) is not getattr(target, name)
    ]

    assert not reimplemented, (
        f"{shim_name} defines its own {reimplemented} instead of re-exporting "
        f"{target_name}'s; the deprecated path can now drift from the supported one"
    )


@pytest.mark.parametrize("shim_name", sorted(SHIMS))
def test_shim_warns_on_import(shim_name):
    importlib.import_module(shim_name)  # ensure it is in sys.modules
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(importlib.import_module(shim_name))

    assert any(issubclass(w.category, DeprecationWarning) for w in caught), (
        f"{shim_name} is deprecated but importing it says nothing"
    )
