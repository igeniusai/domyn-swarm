# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pandas as pd

from domyn_swarm.helpers.io import load_dataframe


def test_load_dataframe_supports_brace_ranges(tmp_path: Path) -> None:
    for i in range(1, 4):
        df = pd.DataFrame({"x": [i]})
        df.to_parquet(tmp_path / f"file-00{i:04d}.parquet", index=False)

    pattern = tmp_path / "file-00{0001..0003}.parquet"
    out = load_dataframe(pattern)

    assert out["x"].tolist() == [1, 2, 3]


def test_load_dataframe_supports_brace_ranges_with_wildcards(tmp_path: Path) -> None:
    for shard in (1, 2):
        for suffix in ("a", "b"):
            df = pd.DataFrame({"k": [f"{shard}{suffix}"]})
            df.to_parquet(tmp_path / f"file_{shard}_{suffix}.parquet", index=False)

    pattern = tmp_path / "file_{1..2}_*.parquet"
    out = load_dataframe(pattern)

    assert sorted(out["k"].tolist()) == ["1a", "1b", "2a", "2b"]
