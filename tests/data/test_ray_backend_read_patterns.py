# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
import sys
import types

from domyn_swarm.data.backends.ray_backend import RayBackend


class _FakeDataset:
    def __init__(self, arg):
        self.arg = arg
        self.limited: int | None = None

    def limit(self, n: int):
        self.limited = n
        return self


def test_ray_backend_read_expands_brace_ranges(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def _read_parquet(arg, **_kwargs):
        seen["arg"] = arg
        return _FakeDataset(arg)

    ray_mod = types.ModuleType("ray")
    ray_data_mod = types.ModuleType("ray.data")
    ray_data_mod.read_parquet = _read_parquet
    ray_mod.data = ray_data_mod

    monkeypatch.setitem(sys.modules, "ray", ray_mod)
    monkeypatch.setitem(sys.modules, "ray.data", ray_data_mod)

    backend = RayBackend()
    ds = backend.read(Path("/tmp/file-{0001..0003}.parquet"), limit=2)

    assert isinstance(seen["arg"], list)
    assert seen["arg"] == [
        "/tmp/file-0001.parquet",
        "/tmp/file-0002.parquet",
        "/tmp/file-0003.parquet",
    ]
    assert ds.limited == 2
