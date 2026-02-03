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


def test_arrow_shard_store_finalize_includes_base_when_new_parts(tmp_path):
    base = tmp_path / "ckpt.parquet"
    store = ArrowShardStore(base.as_posix())
    table = pa.Table.from_pydict({"_row_id": [1, 2], "text": ["a", "b"]})
    store.prepare(table, "_row_id")
    asyncio.run(store.flush(FlushBatch(ids=[1, 2], rows=["x", "y"]), output_cols=["out"]))
    store.finalize()

    parts_dir = tmp_path / "ckpt"
    for part in parts_dir.glob("part-*.parquet"):
        part.unlink()

    store2 = ArrowShardStore(base.as_posix())
    asyncio.run(store2.flush(FlushBatch(ids=[3], rows=["z"]), output_cols=["out"]))
    merged = store2.finalize()

    merged_dict = merged.to_pydict()
    ids = merged_dict.get("_row_id", [])
    outs = merged_dict.get("out", [])
    by_id = {ids[i]: outs[i] for i in range(len(ids))}
    assert by_id[1] == "x"
    assert by_id[2] == "y"
    assert by_id[3] == "z"


def test_arrow_shard_store_finalize_retries_with_large_offsets_on_overflow(tmp_path, monkeypatch):
    base = tmp_path / "ckpt.parquet"
    store = ArrowShardStore(base.as_posix())
    table = pa.Table.from_pydict({"_row_id": [1, 2], "text": ["a", "b"]})
    store.prepare(table, "_row_id")
    asyncio.run(store.flush(FlushBatch(ids=[1, 2], rows=["x", "y"]), output_cols=["out"]))

    import domyn_swarm.checkpoint.arrow_store as arrow_store

    real_take = arrow_store.pc.take
    call_count = {"n": 0}

    def fake_take(data, indices, *args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise pa.ArrowInvalid(
                "offset overflow while concatenating arrays, consider casting input from `string` "
                "to `large_string` first."
            )
        assert isinstance(data, pa.Table)
        assert pa.types.is_large_string(data.schema.field("out").type)
        return real_take(data, indices, *args, **kwargs)

    monkeypatch.setattr(arrow_store.pc, "take", fake_take)
    monkeypatch.setattr(arrow_store, "_normalize_tables_for_concat", lambda tables, id_col: tables)

    merged = store.finalize()
    assert merged.num_rows == 2
    assert pa.types.is_large_string(merged.schema.field("out").type)


def test_arrow_shard_store_finalize_promotes_string_to_large_string_before_concat(tmp_path):
    base = tmp_path / "ckpt.parquet"
    store = ArrowShardStore(base.as_posix())

    base_table = pa.Table.from_pydict({"_row_id": [1], "reasoning_trace": ["a"]})
    with store.fs.open(store.base_path, "wb") as f:
        import pyarrow.parquet as pq

        pq.write_table(base_table, f, use_dictionary=False)

    parts_dir = tmp_path / "ckpt"
    parts_dir.mkdir(exist_ok=True)
    part_table = pa.Table.from_arrays(
        [pa.array([2], type=pa.int64()), pa.array(["b"], type=pa.large_string())],
        names=["_row_id", "reasoning_trace"],
    )
    part_path = parts_dir / "part-000.parquet"
    with store.fs.open(part_path.as_posix(), "wb") as f:
        import pyarrow.parquet as pq

        pq.write_table(part_table, f, use_dictionary=False)

    merged = store.finalize()
    assert merged.num_rows == 2
    assert pa.types.is_large_string(merged.schema.field("reasoning_trace").type)
