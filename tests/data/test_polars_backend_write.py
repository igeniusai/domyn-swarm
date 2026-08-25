# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from domyn_swarm.data.backends.polars_backend import PolarsBackend


def test_polars_write_accepts_nshards_one(tmp_path: Path) -> None:
    pl = pytest.importorskip("polars")
    backend = PolarsBackend()

    out = tmp_path / "out.parquet"
    backend.write(pl.DataFrame({"a": [1, 2]}), out, nshards=1)

    assert out.exists()
    assert pl.read_parquet(out).to_dict(as_series=False) == {"a": [1, 2]}


def test_polars_write_sharded_to_directory(tmp_path: Path) -> None:
    pl = pytest.importorskip("polars")
    backend = PolarsBackend()

    out_dir = tmp_path / "out"  # suffixless path => treat as dataset directory
    backend.write(pl.DataFrame({"a": list(range(10))}), out_dir, nshards=3)

    parts = sorted(out_dir.glob("data-*.parquet"))
    assert len(parts) == 3

    values: list[int] = []
    for part in parts:
        values.extend(pl.read_parquet(part)["a"].to_list())
    assert sorted(values) == list(range(10))
