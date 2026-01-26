import asyncio

import pandas as pd

from domyn_swarm.checkpoint.store import FlushBatch, InMemoryStore, ParquetShardStore


def test_in_memory_store_roundtrip():
    """Stores outputs in memory and returns a DataFrame."""
    store = InMemoryStore()
    df = pd.DataFrame({"_row_id": [1, 2], "text": ["a", "b"]})
    store.prepare(df, "_row_id")
    asyncio.run(store.flush(FlushBatch(ids=[1, 2], rows=["x", "y"]), output_cols=["out"]))
    out = store.finalize()
    assert "out" in out.columns
    assert out.loc[1, "out"] == "x"


def test_parquet_shard_store_flush_and_finalize(tmp_path):
    """Writes shard outputs to disk and merges them."""
    base = tmp_path / "ckpt.parquet"
    store = ParquetShardStore(base.as_posix())
    df = pd.DataFrame({"_row_id": [1, 2], "text": ["a", "b"]})
    store.prepare(df, "_row_id")
    asyncio.run(store.flush(FlushBatch(ids=[1, 2], rows=["x", "y"]), output_cols=["out"]))
    out = store.finalize()
    assert "out" in out.columns
    assert (tmp_path / "ckpt.parquet").exists()
