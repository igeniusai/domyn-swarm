# Copyright 2025 iGenius S.p.A
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import pytest

from domyn_swarm.checkpoint.store import FlushBatch, ParquetShardStore


@pytest.mark.asyncio
async def test_parquet_shard_store_roundtrip_and_dedup(tmp_path):
    base = tmp_path / "run.parquet"
    store = ParquetShardStore(f"file://{base}")

    df = pd.DataFrame({"_row_id": [0, 1, 2], "messages": ["a", "b", "c"]})
    todo = store.prepare(df, "_row_id")
    assert todo.shape[0] == 3

    await store.flush(FlushBatch(ids=[0, 1], rows=["x", "y"]), output_cols=["result"])
    await store.flush(FlushBatch(ids=[1], rows=["y2"]), output_cols=["result"])

    out = store.finalize()
    assert out.index.name == "_row_id"
    assert out.loc[0, "result"] == "x"
    assert out.loc[1, "result"] == "y2"


@pytest.mark.asyncio
async def test_parquet_shard_store_finalize_includes_base_when_new_parts(tmp_path):
    base = tmp_path / "run.parquet"
    store = ParquetShardStore(f"file://{base}")

    await store.flush(FlushBatch(ids=[0, 1], rows=["x", "y"]), output_cols=["result"])
    store.finalize()

    parts_dir = tmp_path / "run"
    for part in parts_dir.glob("part-*.parquet"):
        part.unlink()

    store2 = ParquetShardStore(f"file://{base}")
    await store2.flush(FlushBatch(ids=[2], rows=["z"]), output_cols=["result"])
    out = store2.finalize()

    assert out.loc[0, "result"] == "x"
    assert out.loc[1, "result"] == "y"
    assert out.loc[2, "result"] == "z"
