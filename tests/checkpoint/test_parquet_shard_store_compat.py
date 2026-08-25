# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import pandas as pd

from domyn_swarm.checkpoint.store import ParquetShardStore


def test_finalize_reads_legacy_index_column_as_id(tmp_path):
    base = tmp_path / "legacy.parquet"

    # Simulate a legacy parquet that has the id in the parquet index column
    # (pandas often writes this as '__index_level_0__' when the index is unnamed).
    df = pd.DataFrame({"result": ["a", "b"]}, index=[0, 1])
    df.to_parquet(base)

    store = ParquetShardStore(f"file://{base}")
    store.id_col = "_row_id"
    out = store.finalize()

    assert out.index.name == "_row_id"
    assert out.loc[0, "result"] == "a"
    assert out.loc[1, "result"] == "b"
