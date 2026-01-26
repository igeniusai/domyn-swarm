import asyncio

import pyarrow as pa

from domyn_swarm.checkpoint.arrow_store import ArrowShardStore, InMemoryArrowStore
from domyn_swarm.checkpoint.store import FlushBatch


def test_in_memory_arrow_store_roundtrip():
    """Stores outputs in memory and returns an Arrow table."""
    table = pa.Table.from_pydict({"_row_id": [1, 2], "text": ["a", "b"]})
    store = InMemoryArrowStore()
    store.prepare(table, "_row_id")
    asyncio.run(store.flush(FlushBatch(ids=[1, 2], rows=["x", "y"]), output_cols=["out"]))
    out = store.finalize()
    assert out.num_rows == 2
    assert "out" in out.column_names


def test_arrow_shard_store_flush_and_finalize(tmp_path):
    """Writes shard outputs and merges them on finalize."""
    base = tmp_path / "ckpt.parquet"
    store = ArrowShardStore(base.as_posix())
    table = pa.Table.from_pydict({"_row_id": [1, 2], "text": ["a", "b"]})
    store.prepare(table, "_row_id")
    asyncio.run(store.flush(FlushBatch(ids=[1, 2], rows=["x", "y"]), output_cols=["out"]))
    merged = store.finalize()
    assert merged.num_rows == 2
    assert (tmp_path / "ckpt.parquet").exists()
