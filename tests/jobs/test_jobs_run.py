# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import sys
import tempfile
import types
from unittest.mock import patch

import pandas as pd
import pytest

from domyn_swarm.jobs import SwarmJob
from domyn_swarm.jobs.api.base import OutputJoinMode
import domyn_swarm.jobs.cli.run as run_mod
from domyn_swarm.jobs.cli.run import (
    _amain,
    _load_cls,
    build_job_from_args,
    parse_args,
    run_job_unified,
)


class DummySwarmJob(SwarmJob):
    max_concurrency = 2
    retries = 1
    timeout = 10
    output_mode = OutputJoinMode.APPEND

    def __init__(self, **kwargs):
        kwargs.setdefault("endpoint", "http://dummy-endpoint")
        kwargs.setdefault("model", "dummy-model")
        kwargs.setdefault("input_column_name", "messages")
        kwargs.setdefault("output_cols", "output")
        super().__init__(**kwargs)
        self.params = kwargs
        self.output_mode = kwargs.get("output_mode", OutputJoinMode.APPEND)

    async def transform_items(self, items: list):
        return [f"test_shard_{i}" for i in items]


class DummyDictSwarmJob(SwarmJob):
    output_mode = OutputJoinMode.APPEND

    def __init__(self, **kwargs):
        kwargs.setdefault("endpoint", "http://dummy-endpoint")
        kwargs.setdefault("model", "dummy-model")
        kwargs.setdefault("input_column_name", "messages")
        kwargs.setdefault("output_cols", "output")
        kwargs.setdefault("default_output_cols", [])
        super().__init__(**kwargs)
        self.params = kwargs
        self.output_mode = kwargs.get("output_mode", OutputJoinMode.APPEND)

    async def transform_items(self, items: list):
        return [{"output": f"test_shard_{i}"} for i in items]


def test_load_cls_runtime():
    mod = types.ModuleType("fake_module")

    class Dummy:
        pass

    mod.Dummy = Dummy
    sys.modules["fake_module"] = mod

    loaded = _load_cls("fake_module:Dummy")
    assert loaded is Dummy


def test_build_job_from_args(monkeypatch):
    monkeypatch.setenv("JOB_CLASS", "domyn_swarm.jobs.cli.run:DummySwarmJob")
    monkeypatch.setenv("MODEL", "gpt-4")
    monkeypatch.setenv("ENDPOINT", "http://localhost")

    monkeypatch.setattr(run_mod, "_load_cls", lambda path: DummySwarmJob)

    class Args:
        job_class = None
        model = None
        endpoint = None
        job_kwargs = '{"some_param": "value"}'

    job_cls, job_kwargs = build_job_from_args(Args())
    assert job_cls == DummySwarmJob
    assert job_kwargs["some_param"] == "value"
    assert job_kwargs["model"] == "gpt-4"
    assert job_kwargs["endpoint"] == "http://localhost"


@pytest.mark.asyncio
async def test_run_job_unified_streaming():
    df = pd.DataFrame({"messages": [1, 2, 3, 4]})
    out_df = await run_job_unified(
        DummySwarmJob,
        df,
        input_col="messages",
        output_cols=["output"],
        store_uri="file://" + tempfile.mkdtemp() + "/out.parquet",
        nshards=2,
    )
    assert "output" in out_df.columns
    assert out_df.shape[0] == 4
    assert out_df["output"].str.startswith("test_shard").all()


@pytest.mark.asyncio
async def test_run_job_unified_requires_id_column(tmp_path):
    df = pd.DataFrame({"doc_id": [10, 11], "messages": [1, 2]})
    out_df = await run_job_unified(
        lambda: DummySwarmJob(id_column_name="doc_id"),
        df,
        input_col="messages",
        output_cols=["output"],
        store_uri=f"file://{tmp_path / 'out.parquet'}",
    )
    assert out_df["doc_id"].tolist() == [10, 11]
    assert out_df["output"].str.startswith("test_shard").all()


@pytest.mark.asyncio
async def test_run_job_unified_missing_id_column_raises(tmp_path):
    df = pd.DataFrame({"messages": [1, 2]})
    with pytest.raises(ValueError, match="id column"):
        await run_job_unified(
            lambda: DummySwarmJob(id_column_name="doc_id"),
            df,
            input_col="messages",
            output_cols=["output"],
            store_uri=f"file://{tmp_path / 'out.parquet'}",
        )


@pytest.mark.asyncio
async def test_run_job_unified_ray_requires_user_id_column():
    """Ray backend requires a user-provided id column for resume semantics."""
    with pytest.raises(ValueError, match="requires a user-provided id column"):
        await run_job_unified(
            DummySwarmJob,
            data={},
            input_col="messages",
            output_cols=["output"],
            data_backend="ray",
        )


@pytest.mark.asyncio
async def test_run_job_unified_forwards_ray_address(monkeypatch):
    """Forward ray address from run_job_unified into the ray runner."""
    df = pd.DataFrame({"doc_id": [1, 2], "messages": [1, 2]})
    seen: dict[str, object] = {}

    async def _fake_run_ray_job(*args, **kwargs):
        seen.update(kwargs)
        return "ok"

    monkeypatch.setattr("domyn_swarm.jobs.execution.dispatch.run_ray_job", _fake_run_ray_job)
    monkeypatch.setattr(
        "domyn_swarm.jobs.execution.dispatch.get_backend",
        lambda name: types.SimpleNamespace(name="ray"),
    )

    out = await run_job_unified(
        lambda: DummySwarmJob(id_column_name="doc_id"),
        df,
        input_col="messages",
        output_cols=["output"],
        data_backend="ray",
        ray_address="ray://head:10001",
    )
    assert out == "ok"
    assert seen["ray_address"] == "ray://head:10001"


@pytest.mark.asyncio
async def test_run_job_unified_arrow_runner_id_column(tmp_path):
    df = pd.DataFrame({"doc_id": [10, 11], "messages": [1, 2]})
    out_df = await run_job_unified(
        lambda: DummySwarmJob(id_column_name="doc_id"),
        df,
        input_col="messages",
        output_cols=["output"],
        store_uri=f"file://{tmp_path / 'out.parquet'}",
        runner="arrow",
    )
    assert out_df["doc_id"].tolist() == [10, 11]
    assert out_df["output"].tolist() == ["test_shard_1", "test_shard_2"]


@pytest.mark.asyncio
async def test_run_job_unified_polars_runner_lazy(monkeypatch, tmp_path):
    """Run the Arrow-backed polars path with LazyFrame input.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Pytest temporary directory.
    """
    pl = pytest.importorskip("polars")
    collect_calls: dict[str, int] = {"collect": 0, "collect_batches": 0}
    orig_collect = pl.LazyFrame.collect
    orig_collect_batches = pl.LazyFrame.collect_batches

    def _collect_wrapper(self, *args, **kwargs):
        collect_calls["collect"] += 1
        return orig_collect(self, *args, **kwargs)

    def _collect_batches_wrapper(self, *args, **kwargs):
        collect_calls["collect_batches"] += 1
        return orig_collect_batches(self, *args, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, "collect", _collect_wrapper)
    monkeypatch.setattr(pl.LazyFrame, "collect_batches", _collect_batches_wrapper)

    data = pl.DataFrame({"doc_id": [10, 11], "messages": [1, 2]}).lazy()
    out_df = await run_job_unified(
        lambda: DummySwarmJob(id_column_name="doc_id"),
        data,
        input_col="messages",
        output_cols=["output"],
        store_uri=f"file://{tmp_path / 'out.parquet'}",
        data_backend="polars",
        runner="arrow",
    )
    assert isinstance(out_df, pl.DataFrame)
    assert out_df["doc_id"].to_list() == [10, 11]
    assert out_df["output"].to_list() == ["test_shard_1", "test_shard_2"]
    assert collect_calls["collect_batches"] >= 1
    # Some polars versions call LazyFrame.collect within collect_batches; keep this bounded.
    assert collect_calls["collect"] <= 2


@pytest.mark.asyncio
async def test_run_job_unified_shard_mode_id_resume_with_input_change(tmp_path):
    store_uri = f"file://{tmp_path / 'out.parquet'}"
    data = pd.DataFrame({"messages": list(range(6))})

    await run_job_unified(
        DummySwarmJob,
        data.head(4),
        input_col="messages",
        output_cols=["output"],
        store_uri=store_uri,
        nshards=2,
        shard_mode="id",
    )

    class GuardJob(DummySwarmJob):
        def __init__(self, cutoff: int, **kwargs):
            super().__init__(**kwargs)
            self.cutoff = cutoff

        async def transform_items(self, items: list):
            if any(i < self.cutoff for i in items):
                raise RuntimeError("unexpected recompute")
            return [f"test_shard_{i}" for i in items]

    out_df = await run_job_unified(
        lambda: GuardJob(cutoff=4),
        data,
        input_col="messages",
        output_cols=["output"],
        store_uri=store_uri,
        nshards=2,
        shard_mode="id",
    )
    assert out_df["messages"].tolist() == list(range(6))
    assert out_df["output"].tolist() == [f"test_shard_{i}" for i in range(6)]


@pytest.mark.asyncio
async def test_run_job_unified_global_resume_with_index_sharding(tmp_path):
    store_uri = f"file://{tmp_path / 'out.parquet'}"
    data = pd.DataFrame({"messages": list(range(6))})

    await run_job_unified(
        DummySwarmJob,
        data.head(4),
        input_col="messages",
        output_cols=["output"],
        store_uri=store_uri,
        nshards=2,
        shard_mode="index",
    )

    class GuardJob(DummySwarmJob):
        def __init__(self, cutoff: int, **kwargs):
            super().__init__(**kwargs)
            self.cutoff = cutoff

        async def transform_items(self, items: list):
            if any(i < self.cutoff for i in items):
                raise RuntimeError("unexpected recompute")
            return [f"test_shard_{i}" for i in items]

    out_df = await run_job_unified(
        lambda: GuardJob(cutoff=4),
        data,
        input_col="messages",
        output_cols=["output"],
        store_uri=store_uri,
        nshards=2,
        shard_mode="index",
        global_resume=True,
    )
    assert out_df["messages"].tolist() == list(range(6))
    assert out_df["output"].tolist() == [f"test_shard_{i}" for i in range(6)]


@pytest.mark.asyncio
async def test_run_job_unified_global_resume_arrow_runner(tmp_path):
    store_uri = f"file://{tmp_path / 'out.parquet'}"
    data = pd.DataFrame({"messages": list(range(6))})

    await run_job_unified(
        DummySwarmJob,
        data.head(4),
        input_col="messages",
        output_cols=["output"],
        store_uri=store_uri,
        nshards=2,
        shard_mode="index",
        runner="arrow",
    )

    class GuardJob(DummySwarmJob):
        def __init__(self, cutoff: int, **kwargs):
            super().__init__(**kwargs)
            self.cutoff = cutoff

        async def transform_items(self, items: list):
            if any(i < self.cutoff for i in items):
                raise RuntimeError("unexpected recompute")
            return [f"test_shard_{i}" for i in items]

    out_df = await run_job_unified(
        lambda: GuardJob(cutoff=4),
        data,
        input_col="messages",
        output_cols=["output"],
        store_uri=store_uri,
        nshards=2,
        shard_mode="index",
        runner="arrow",
        global_resume=True,
    )
    assert out_df["messages"].tolist() == list(range(6))
    assert out_df["output"].tolist() == [f"test_shard_{i}" for i in range(6)]


@pytest.mark.asyncio
async def test_run_job_unified_global_resume_with_custom_id_column_pandas(tmp_path):
    """Regression: global resume must not lose a custom id column when finalizing.

    `_finalize_global_resume` used to construct a fresh `ParquetShardStore` and call
    `finalize()` on it without ever calling `prepare()`, so the store's id column
    stayed at its default (`_row_id`) instead of the job's real id column. `finalize()`
    then looked for `_row_id` in a parquet shard that only had the custom id column
    (`doc_id`) and raised `ValueError: Merged parquet is missing id column '_row_id'`.

    `shard_mode="index"` (not "id") keeps every shard non-empty so this isolates the
    id-column bug from the separate empty-shard bug.
    """
    store_uri = f"file://{tmp_path / 'out.parquet'}"
    data = pd.DataFrame({"doc_id": [1, 2, 3, 4], "messages": [1, 2, 3, 4]})

    out_df = await run_job_unified(
        lambda: DummySwarmJob(id_column_name="doc_id"),
        data,
        input_col="messages",
        output_cols=["output"],
        store_uri=store_uri,
        nshards=2,
        shard_mode="index",
        global_resume=True,
    )
    assert sorted(out_df["doc_id"].tolist()) == [1, 2, 3, 4]
    assert sorted(out_df["output"].tolist()) == [f"test_shard_{i}" for i in [1, 2, 3, 4]]


@pytest.mark.asyncio
async def test_run_job_unified_global_resume_with_custom_id_column_arrow(tmp_path):
    """Arrow-engine counterpart of the pandas custom-id-column global resume bug.

    `_finalize_arrow_global_resume` -> `_merge_shard_outputs` had the same defect:
    a fresh `ArrowShardStore` was finalized without ever calling `prepare()`, so it
    looked for the default `_row_id` instead of the job's real id column.
    """
    store_uri = f"file://{tmp_path / 'out.parquet'}"
    data = pd.DataFrame({"doc_id": [1, 2, 3, 4], "messages": [1, 2, 3, 4]})

    out_df = await run_job_unified(
        lambda: DummySwarmJob(id_column_name="doc_id"),
        data,
        input_col="messages",
        output_cols=["output"],
        store_uri=store_uri,
        nshards=2,
        shard_mode="index",
        runner="arrow",
        global_resume=True,
    )
    assert sorted(out_df["doc_id"].tolist()) == [1, 2, 3, 4]
    assert sorted(out_df["output"].tolist()) == [f"test_shard_{i}" for i in [1, 2, 3, 4]]


@pytest.mark.asyncio
async def test_run_job_unified_id_sharding_tolerates_empty_shards_pandas(tmp_path):
    """Regression: id-hash sharding can produce empty shards; empty shards must not crash.

    `shard_indices_by_id` hashes ids into buckets by `hash(id) % nshards`. With these
    five ids and four shards, two shards get zero rows
    (`shard_indices_by_id(pd.Series([10, 11, 12, 13, 14]), 4)` gives shard sizes
    `[1, 4, 0, 0]`). `ParquetShardStore.finalize()` used to crash on an empty shard
    with `KeyError: "None of ['doc_id'] are in the columns"` because it fell back to
    `pd.DataFrame().set_index(self.id_col)`, and an empty `pd.DataFrame()` has no
    columns to index by.
    """
    store_uri = f"file://{tmp_path / 'out.parquet'}"
    data = pd.DataFrame({"doc_id": [10, 11, 12, 13, 14], "messages": [1, 2, 3, 4, 5]})

    out_df = await run_job_unified(
        lambda: DummySwarmJob(id_column_name="doc_id"),
        data,
        input_col="messages",
        output_cols=["output"],
        store_uri=store_uri,
        nshards=4,
        shard_mode="id",
    )
    assert sorted(out_df["doc_id"].tolist()) == [10, 11, 12, 13, 14]
    assert sorted(out_df["output"].tolist()) == [f"test_shard_{i}" for i in [1, 2, 3, 4, 5]]


@pytest.mark.asyncio
async def test_run_job_unified_id_sharding_tolerates_empty_shards_arrow(tmp_path):
    """Arrow-engine counterpart: an empty id-hash shard used to break `pa.concat_tables`.

    A shard with zero assigned rows produced a result table with a null-typed,
    "output"-less schema (built from empty Python lists via `pa.Table.from_pydict`),
    which didn't match the other shards' schemas and made `pa.concat_tables(parts)`
    raise `pyarrow.lib.ArrowInvalid: Schema at index N was different`.
    """
    store_uri = f"file://{tmp_path / 'out.parquet'}"
    data = pd.DataFrame({"doc_id": [10, 11, 12, 13, 14], "messages": [1, 2, 3, 4, 5]})

    out_df = await run_job_unified(
        lambda: DummySwarmJob(id_column_name="doc_id"),
        data,
        input_col="messages",
        output_cols=["output"],
        store_uri=store_uri,
        nshards=4,
        shard_mode="id",
        runner="arrow",
    )
    assert sorted(out_df["doc_id"].tolist()) == [10, 11, 12, 13, 14]
    assert sorted(out_df["output"].tolist()) == [f"test_shard_{i}" for i in [1, 2, 3, 4, 5]]


@pytest.mark.asyncio
@pytest.mark.parametrize("global_resume", [False, True])
async def test_run_job_unified_id_sharding_tolerates_empty_shards_polars(tmp_path, global_resume):
    """Polars-engine counterpart: an empty id-hash shard broke two things in turn.

    Polars shares the Arrow checkpoint stores, so an empty shard's `ArrowShardStore`
    result is the same null-typed, id-only table as the plain Arrow engine's. That
    first broke `PolarsJobRunner._finalize_output`'s `base_df.join(out_df, ...)` with
    `SchemaError: datatypes of join keys don't match` (the id column's `Null` dtype
    vs. the input's `Int64`). After aligning that dtype, the per-shard joined result
    was still missing the output column entirely (nothing to derive it from), which
    broke `pl.concat(parts, how="vertical")` in `_run_polars_sharded` with
    `ShapeError: unable to append to a DataFrame of width 3 with a DataFrame of
    width 2`. `_run_polars_sharded` runs (and therefore can raise) regardless of
    `global_resume`, since its concatenated return value is only used when
    `global_resume` is False -- hence parametrizing both here.
    """
    pytest.importorskip("polars")
    store_uri = f"file://{tmp_path / 'out.parquet'}"
    data = pd.DataFrame({"doc_id": [10, 11, 12, 13, 14], "messages": [1, 2, 3, 4, 5]})

    out_df = await run_job_unified(
        lambda: DummySwarmJob(id_column_name="doc_id"),
        data,
        input_col="messages",
        output_cols=["output"],
        store_uri=store_uri,
        nshards=4,
        shard_mode="id",
        runner="arrow",
        data_backend="polars",
        global_resume=global_resume,
    )
    assert sorted(out_df["doc_id"].to_list()) == [10, 11, 12, 13, 14]
    assert sorted(out_df["output"].to_list()) == [f"test_shard_{i}" for i in [1, 2, 3, 4, 5]]
    assert str(out_df["doc_id"].dtype) == "Int64"


@pytest.mark.asyncio
async def test_run_job_unified_global_resume_polars_runner(tmp_path):
    pytest.importorskip("polars")

    import polars as pl

    store_uri = f"file://{tmp_path / 'out.parquet'}"
    data = pl.DataFrame({"messages": list(range(6))})

    await run_job_unified(
        DummySwarmJob,
        data.head(4),
        input_col="messages",
        output_cols=["output"],
        store_uri=store_uri,
        nshards=2,
        shard_mode="index",
        data_backend="polars",
        runner="arrow",
    )

    class GuardJob(DummySwarmJob):
        def __init__(self, cutoff: int, **kwargs):
            super().__init__(**kwargs)
            self.cutoff = cutoff

        async def transform_items(self, items: list):
            if any(i < self.cutoff for i in items):
                raise RuntimeError("unexpected recompute")
            return [f"test_shard_{i}" for i in items]

    out_df = await run_job_unified(
        lambda: GuardJob(cutoff=4),
        data,
        input_col="messages",
        output_cols=["output"],
        store_uri=store_uri,
        nshards=2,
        shard_mode="index",
        data_backend="polars",
        runner="arrow",
        global_resume=True,
    )
    assert out_df["messages"].to_list() == list(range(6))
    assert out_df["output"].to_list() == [f"test_shard_{i}" for i in range(6)]


@pytest.mark.asyncio
async def test_run_job_unified_polars_runner_lazy_resume_skips_done_ids(tmp_path):
    pytest.importorskip("polars")

    import polars as pl

    store_uri = f"file://{tmp_path / 'out.parquet'}"
    count = 100

    data = pl.DataFrame(
        {"doc_id": list(range(1000, 1000 + count)), "messages": list(range(count))}
    ).lazy()
    out_df = await run_job_unified(
        lambda: DummySwarmJob(id_column_name="doc_id"),
        data,
        input_col="messages",
        output_cols=["output"],
        store_uri=store_uri,
        data_backend="polars",
        runner="arrow",
        checkpoint_every=1,
    )
    assert out_df["doc_id"].to_list() == list(range(1000, 1000 + count))
    assert out_df["output"].to_list() == [f"test_shard_{i}" for i in range(count)]

    class ExplodingSwarmJob(DummySwarmJob):
        async def transform_items(self, items: list):
            raise RuntimeError("unexpected recompute")

    data = pl.DataFrame(
        {"doc_id": list(range(1000, 1000 + count)), "messages": list(range(count))}
    ).lazy()
    out_df = await run_job_unified(
        lambda: ExplodingSwarmJob(id_column_name="doc_id"),
        data,
        input_col="messages",
        output_cols=["output"],
        store_uri=store_uri,
        data_backend="polars",
        runner="arrow",
        checkpoint_every=1,
    )
    assert out_df["doc_id"].to_list() == list(range(1000, 1000 + count))
    assert out_df["output"].to_list() == [f"test_shard_{i}" for i in range(count)]


@pytest.mark.asyncio
async def test_run_job_unified_polars_runner_lazy_resume_without_id_column(tmp_path):
    pytest.importorskip("polars")

    import polars as pl

    store_uri = f"file://{tmp_path / 'out.parquet'}"
    count = 100

    data = pl.DataFrame({"messages": list(range(count))}).lazy()
    out_df = await run_job_unified(
        DummySwarmJob,
        data,
        input_col="messages",
        output_cols=["output"],
        store_uri=store_uri,
        data_backend="polars",
        runner="arrow",
        checkpoint_every=1,
    )
    assert out_df["_row_id"].to_list() == list(range(count))
    assert out_df["output"].to_list() == [f"test_shard_{i}" for i in range(count)]

    class ExplodingSwarmJob(DummySwarmJob):
        async def transform_items(self, items: list):
            raise RuntimeError("unexpected recompute")

    data = pl.DataFrame({"messages": list(range(count))}).lazy()
    out_df = await run_job_unified(
        ExplodingSwarmJob,
        data,
        input_col="messages",
        output_cols=["output"],
        store_uri=store_uri,
        data_backend="polars",
        runner="arrow",
        checkpoint_every=1,
    )
    assert out_df["_row_id"].to_list() == list(range(count))
    assert out_df["output"].to_list() == [f"test_shard_{i}" for i in range(count)]


@pytest.mark.asyncio
async def test_run_job_unified_polars_runner_lazy_output_dir_streams(tmp_path):
    """Stream LazyFrame output directly to an output directory.

    This is the memory-stable path for large outputs: Polars joins the input LazyFrame with
    checkpoint outputs and writes a single shard into the output directory.

    Args:
        tmp_path: Pytest temporary directory.
    """
    pytest.importorskip("polars")

    import polars as pl

    output_dir = tmp_path / "out_dir"
    store_uri = f"file://{tmp_path / 'out.parquet'}"

    data = pl.DataFrame({"messages": [1, 2]}).lazy()
    result = await run_job_unified(
        lambda: DummySwarmJob(),
        data,
        input_col="messages",
        output_cols=["output"],
        store_uri=store_uri,
        data_backend="polars",
        runner="arrow",
        output_path=output_dir,
    )

    assert result is None
    shard = output_dir / "data-000000.parquet"
    assert shard.exists()

    df_out = pd.read_parquet(shard).sort_values("_row_id")
    assert df_out["messages"].tolist() == [1, 2]
    assert df_out["output"].tolist() == ["test_shard_1", "test_shard_2"]


@pytest.mark.asyncio
async def test_run_job_unified_polars_runner_lazy_output_dir_shard_output(tmp_path):
    """Write one parquet file per shard into an output directory.

    This uses checkpoint outputs as the source of truth, and materializes one file per shard
    (based on `nshards`).

    Args:
        tmp_path: Pytest temporary directory.
    """
    pytest.importorskip("polars")

    import polars as pl

    output_dir = tmp_path / "out_dir"
    store_uri = f"file://{tmp_path / 'out.parquet'}"

    data = pl.DataFrame({"messages": [1, 2, 3, 4]}).lazy()
    result = await run_job_unified(
        lambda: DummySwarmJob(),
        data,
        input_col="messages",
        output_cols=["output"],
        nshards=2,
        store_uri=store_uri,
        data_backend="polars",
        runner="arrow",
        output_path=output_dir,
        shard_output=True,
    )

    assert result is None
    shard0 = output_dir / "data-0.parquet"
    shard1 = output_dir / "data-1.parquet"
    assert shard0.exists()
    assert shard1.exists()

    df_out = pd.concat(
        [pd.read_parquet(shard0), pd.read_parquet(shard1)],
        ignore_index=True,
    ).sort_values("_row_id")
    assert df_out["messages"].tolist() == [1, 2, 3, 4]
    assert df_out["output"].tolist() == [
        "test_shard_1",
        "test_shard_2",
        "test_shard_3",
        "test_shard_4",
    ]


@pytest.mark.asyncio
async def test_arrow_runner_output_modes_pandas(tmp_path):
    """Validate Arrow runner output modes for pandas backend.

    Args:
        tmp_path: Pytest temporary directory.
    """
    df = pd.DataFrame({"messages": [1, 2], "extra": ["a", "b"]})
    expected_cols = {
        OutputJoinMode.APPEND: {"_row_id", "messages", "extra", "output"},
        OutputJoinMode.IO_ONLY: {"_row_id", "messages", "output"},
        OutputJoinMode.REPLACE: {"_row_id", "output"},
    }
    for mode, cols in expected_cols.items():
        out_df = await run_job_unified(
            lambda m=mode: DummySwarmJob(output_mode=m),
            df,
            input_col="messages",
            output_cols=["output"],
            store_uri=f"file://{tmp_path / f'out_{mode.value}.parquet'}",
            runner="arrow",
        )
        assert set(out_df.columns) == cols


@pytest.mark.asyncio
async def test_arrow_runner_output_modes_polars(tmp_path):
    """Validate Arrow runner output modes for polars backend.

    Args:
        tmp_path: Pytest temporary directory.
    """
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"messages": [1, 2], "extra": ["a", "b"]})
    expected_cols = {
        OutputJoinMode.APPEND: {"_row_id", "messages", "extra", "output"},
        OutputJoinMode.IO_ONLY: {"_row_id", "messages", "output"},
        OutputJoinMode.REPLACE: {"_row_id", "output"},
    }
    for mode, cols in expected_cols.items():
        out_df = await run_job_unified(
            lambda m=mode: DummySwarmJob(output_mode=m),
            df,
            input_col="messages",
            output_cols=["output"],
            store_uri=f"file://{tmp_path / f'out_{mode.value}.parquet'}",
            data_backend="polars",
            runner="arrow",
        )
        assert isinstance(out_df, pl.DataFrame)
        assert set(out_df.columns) == cols


@pytest.mark.asyncio
async def test_arrow_runner_uses_pandas_index_as_id_when_missing(tmp_path):
    df = pd.DataFrame({"messages": [1, 2]}, index=[10, 20])
    out_df = await run_job_unified(
        DummySwarmJob,
        df,
        input_col="messages",
        output_cols=["output"],
        store_uri=f"file://{tmp_path / 'out.parquet'}",
        runner="arrow",
    )
    assert out_df["_row_id"].tolist() == [10, 20]


@pytest.mark.asyncio
async def test_arrow_runner_append_suffixes_on_collision(tmp_path):
    df = pd.DataFrame({"messages": [1, 2], "output": ["orig1", "orig2"]})
    out_df = await run_job_unified(
        DummyDictSwarmJob,
        df,
        input_col="messages",
        output_cols=None,
        store_uri=f"file://{tmp_path / 'out.parquet'}",
        runner="arrow",
    )
    assert {"output_x", "output_y"}.issubset(out_df.columns)
    assert out_df["output_x"].tolist() == ["orig1", "orig2"]
    assert out_df["output_y"].tolist() == ["test_shard_1", "test_shard_2"]


@pytest.mark.asyncio
async def test_arrow_runner_io_only_suffixes_on_collision(tmp_path):
    df = pd.DataFrame({"messages": [1, 2], "output": ["orig1", "orig2"]})
    out_df = await run_job_unified(
        lambda: DummyDictSwarmJob(output_mode=OutputJoinMode.IO_ONLY),
        df,
        input_col="messages",
        output_cols=None,
        store_uri=f"file://{tmp_path / 'out.parquet'}",
        runner="arrow",
    )
    assert {"output_x", "output_y"}.issubset(out_df.columns)
    assert out_df["output_x"].tolist() == ["orig1", "orig2"]
    assert out_df["output_y"].tolist() == ["test_shard_1", "test_shard_2"]


@pytest.mark.asyncio
async def test_arrow_runner_polars_uses_pandas_index_as_id_when_missing(tmp_path):
    pytest.importorskip("polars")
    df = pd.DataFrame({"messages": [1, 2]}, index=[10, 20])
    out_df = await run_job_unified(
        DummySwarmJob,
        df,
        input_col="messages",
        output_cols=["output"],
        store_uri=f"file://{tmp_path / 'out.parquet'}",
        data_backend="polars",
        runner="arrow",
    )
    assert out_df["_row_id"].to_list() == [10, 20]


@pytest.mark.asyncio
async def test_arrow_runner_polars_append_suffixes_on_collision(tmp_path):
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"messages": [1, 2], "output": ["orig1", "orig2"]})
    out_df = await run_job_unified(
        DummyDictSwarmJob,
        df,
        input_col="messages",
        output_cols=None,
        store_uri=f"file://{tmp_path / 'out.parquet'}",
        data_backend="polars",
        runner="arrow",
    )
    assert {"output_x", "output_y"}.issubset(out_df.columns)
    assert out_df["output_x"].to_list() == ["orig1", "orig2"]
    assert out_df["output_y"].to_list() == ["test_shard_1", "test_shard_2"]


@pytest.mark.asyncio
async def test_arrow_runner_polars_io_only_suffixes_on_collision(tmp_path):
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"messages": [1, 2], "output": ["orig1", "orig2"]})
    out_df = await run_job_unified(
        lambda: DummyDictSwarmJob(output_mode=OutputJoinMode.IO_ONLY),
        df,
        input_col="messages",
        output_cols=None,
        store_uri=f"file://{tmp_path / 'out.parquet'}",
        data_backend="polars",
        runner="arrow",
    )
    assert {"output_x", "output_y"}.issubset(out_df.columns)
    assert out_df["output_x"].to_list() == ["orig1", "orig2"]
    assert out_df["output_y"].to_list() == ["test_shard_1", "test_shard_2"]


def test_parse_args_minimal():
    args = parse_args(
        [
            "--job-class",
            "mod:Cls",
            "--model",
            "gpt-4",
            "--input-parquet",
            "in.parquet",
            "--output-parquet",
            "out.parquet",
            "--endpoint",
            "http://localhost",
        ]
    )
    assert args.model == "gpt-4"
    assert args.input_parquet.name == "in.parquet"


@pytest.mark.asyncio
async def test_transform_is_not_supported():
    job = DummySwarmJob()
    with pytest.raises(RuntimeError, match="transform\\(df\\) is no longer supported"):
        await job.transform(pd.DataFrame({"messages": ["hi"]}))


@pytest.mark.asyncio
async def test_amain_end_to_end(monkeypatch, tmp_path):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    checkpoint_dir = tmp_path / "checkpoints"

    # Prepare input dataframe
    df_in = pd.DataFrame({"text": ["hello", "world"]})
    df_in.to_parquet(input_path)

    # Patch helpers
    monkeypatch.setenv("JOB_CLASS", "domyn_swarm.jobs.cli.run:DummySwarmJob")
    monkeypatch.setenv("MODEL", "mock-model")
    monkeypatch.setenv("ENDPOINT", "mock-endpoint")
    monkeypatch.setenv("INPUT_PARQUET", str(input_path))
    monkeypatch.setenv("OUTPUT_PARQUET", str(output_path))
    monkeypatch.setenv("JOB_KWARGS", '{"input_column_name": "text", "output_cols": "output"}')

    monkeypatch.setattr(run_mod, "_load_cls", lambda path: DummySwarmJob)

    args = [
        "--num-shards",
        "1",  # Use single-threaded path
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--job-kwargs",
        '{"input_column_name": "text", "output_cols": "output"}',
    ]
    await _amain(args)

    df_out = pd.read_parquet(output_path)
    print(df_out)
    assert "output" in df_out.columns
    assert df_out.shape[0] == 2


@pytest.mark.asyncio
async def test_amain_writes_shards_directly_for_dir_output(monkeypatch, tmp_path):
    input_path = tmp_path / "input.parquet"
    output_dir = tmp_path / "outdir"
    checkpoint_dir = tmp_path / "checkpoints"

    df_in = pd.DataFrame({"text": ["hello", "world", "third", "fourth"]})
    df_in.to_parquet(input_path)

    monkeypatch.setenv("JOB_CLASS", "ddomyn_swarm.jobs.cli.run:DummySwarmJob")
    monkeypatch.setenv("MODEL", "mock-model")
    monkeypatch.setenv("ENDPOINT", "mock-endpoint")
    monkeypatch.setenv("INPUT_PARQUET", str(input_path))
    monkeypatch.setenv("OUTPUT_PARQUET", str(output_dir))
    monkeypatch.setenv("JOB_KWARGS", '{"input_column_name": "text", "output_cols": "output"}')

    monkeypatch.setattr(run_mod, "_load_cls", lambda path: DummySwarmJob)

    args = [
        "--num-shards",
        "2",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--checkpoint-interval",
        "2",
        "--job-kwargs",
        '{"input_column_name": "text", "output_cols": "output"}',
    ]
    await _amain(args)

    shard0 = output_dir / "data-0.parquet"
    shard1 = output_dir / "data-1.parquet"
    assert shard0.exists()
    assert shard1.exists()

    df0 = pd.read_parquet(shard0)
    df1 = pd.read_parquet(shard1)
    got = pd.concat([df0, df1], ignore_index=True).sort_values("_row_id")
    assert got["output"].str.startswith("test_shard_").all()


@pytest.mark.asyncio
async def test_amain_end_to_end_polars_backend(monkeypatch, tmp_path):
    pytest.importorskip("polars")

    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    checkpoint_dir = tmp_path / "checkpoints"

    df_in = pd.DataFrame({"text": ["hello", "world"]})
    df_in.to_parquet(input_path)

    monkeypatch.setenv("JOB_CLASS", "domyn_swarm.jobs.cli.run:DummySwarmJob")
    monkeypatch.setenv("MODEL", "mock-model")
    monkeypatch.setenv("ENDPOINT", "mock-endpoint")
    monkeypatch.setenv("INPUT_PARQUET", str(input_path))
    monkeypatch.setenv("OUTPUT_PARQUET", str(output_path))
    monkeypatch.setenv(
        "JOB_KWARGS",
        '{"input_column_name":"text","output_cols":"output","data_backend":"polars"}',
    )

    monkeypatch.setattr(run_mod, "_load_cls", lambda path: DummySwarmJob)

    args = [
        "--num-shards",
        "1",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--job-kwargs",
        '{"input_column_name":"text","output_cols":"output","data_backend":"polars"}',
    ]
    await _amain(args)

    df_out = pd.read_parquet(output_path)
    assert "output" in df_out.columns
    assert df_out.shape[0] == 2


@pytest.mark.asyncio
async def test_amain_end_to_end_polars_backend_shard_output(monkeypatch, tmp_path):
    """Write polars outputs as a sharded parquet dataset when output is a directory."""
    pytest.importorskip("polars")

    input_path = tmp_path / "input.parquet"
    output_dir = tmp_path / "output_dir"
    checkpoint_dir = tmp_path / "checkpoints"

    df_in = pd.DataFrame({"messages": [1, 2, 3, 4]})
    df_in.to_parquet(input_path)

    monkeypatch.setenv("JOB_CLASS", "domyn_swarm.jobs.cli.run:DummySwarmJob")
    monkeypatch.setenv("MODEL", "mock-model")
    monkeypatch.setenv("ENDPOINT", "mock-endpoint")
    monkeypatch.setenv("INPUT_PARQUET", str(input_path))
    monkeypatch.setenv("OUTPUT_PARQUET", str(output_dir))
    monkeypatch.setenv(
        "JOB_KWARGS",
        '{"output_cols":"output","data_backend":"polars","backend_read_kwargs":{"use_scan":true}}',
    )

    monkeypatch.setattr(run_mod, "_load_cls", lambda path: DummySwarmJob)

    args = [
        "--num-shards",
        "2",
        "--runner",
        "arrow",
        "--shard-output",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--job-kwargs",
        '{"output_cols":"output","data_backend":"polars","backend_read_kwargs":{"use_scan":true}}',
    ]
    await _amain(args)

    shard0 = output_dir / "data-0.parquet"
    shard1 = output_dir / "data-1.parquet"
    assert shard0.exists()
    assert shard1.exists()

    df_out = pd.concat(
        [pd.read_parquet(shard0), pd.read_parquet(shard1)],
        ignore_index=True,
    ).sort_values("_row_id")
    assert df_out["messages"].tolist() == [1, 2, 3, 4]
    assert df_out["output"].tolist() == [
        "test_shard_1",
        "test_shard_2",
        "test_shard_3",
        "test_shard_4",
    ]


@pytest.mark.asyncio
async def test_amain_arrow_runner_pandas_backend(monkeypatch, tmp_path):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    checkpoint_dir = tmp_path / "checkpoints"

    df_in = pd.DataFrame({"text": ["hello", "world"]})
    df_in.to_parquet(input_path)

    monkeypatch.setenv("JOB_CLASS", "domyn_swarm.jobs.cli.run:DummySwarmJob")
    monkeypatch.setenv("MODEL", "mock-model")
    monkeypatch.setenv("ENDPOINT", "mock-endpoint")
    monkeypatch.setenv("INPUT_PARQUET", str(input_path))
    monkeypatch.setenv("OUTPUT_PARQUET", str(output_path))
    monkeypatch.setenv(
        "JOB_KWARGS",
        '{"input_column_name":"text","output_cols":"output","data_backend":"pandas"}',
    )

    monkeypatch.setattr(run_mod, "_load_cls", lambda path: DummySwarmJob)

    args = [
        "--num-shards",
        "1",
        "--runner",
        "arrow",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--job-kwargs",
        '{"input_column_name":"text","output_cols":"output","data_backend":"pandas"}',
    ]
    await _amain(args)

    df_out = pd.read_parquet(output_path)
    assert "output" in df_out.columns
    assert df_out.shape[0] == 2


@pytest.mark.asyncio
async def test_amain_arrow_runner_sharded(monkeypatch, tmp_path):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    checkpoint_dir = tmp_path / "checkpoints"

    df_in = pd.DataFrame({"text": ["hello", "world"]})
    df_in.to_parquet(input_path)

    monkeypatch.setenv("JOB_CLASS", "domyn_swarm.jobs.cli.run:DummySwarmJob")
    monkeypatch.setenv("MODEL", "mock-model")
    monkeypatch.setenv("ENDPOINT", "mock-endpoint")
    monkeypatch.setenv("INPUT_PARQUET", str(input_path))
    monkeypatch.setenv("OUTPUT_PARQUET", str(output_path))
    monkeypatch.setenv(
        "JOB_KWARGS",
        '{"input_column_name":"text","output_cols":"output","data_backend":"pandas"}',
    )

    monkeypatch.setattr(run_mod, "_load_cls", lambda path: DummySwarmJob)

    args = [
        "--num-shards",
        "2",
        "--runner",
        "arrow",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--job-kwargs",
        '{"input_column_name":"text","output_cols":"output","data_backend":"pandas"}',
    ]
    await _amain(args)

    df_out = pd.read_parquet(output_path)
    assert "output" in df_out.columns
    assert df_out.shape[0] == 2


@pytest.mark.asyncio
async def test_amain_no_resume_forces_recompute(monkeypatch, tmp_path):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    checkpoint_dir = tmp_path / "checkpoints"

    df_in = pd.DataFrame({"text": ["hello", "world"]})
    df_in.to_parquet(input_path)

    monkeypatch.setenv("JOB_CLASS", "domyn_swarm.jobs.cli.run:DummySwarmJob")
    monkeypatch.setenv("MODEL", "mock-model")
    monkeypatch.setenv("ENDPOINT", "mock-endpoint")
    monkeypatch.setenv("INPUT_PARQUET", str(input_path))
    monkeypatch.setenv("OUTPUT_PARQUET", str(output_path))
    monkeypatch.setenv("JOB_KWARGS", '{"input_column_name": "text", "output_cols": "output"}')
    monkeypatch.setattr(run_mod, "_load_cls", lambda path: DummySwarmJob)

    # First run creates checkpoints
    await _amain(
        [
            "--num-shards",
            "1",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--job-kwargs",
            '{"input_column_name": "text", "output_cols": "output"}',
        ]
    )

    # Second run should resume and skip work even if transform_items would raise.
    async def _raise_transform_items(self, items: list):
        raise RuntimeError("should not be called when resuming")

    monkeypatch.setattr(DummySwarmJob, "transform_items", _raise_transform_items)
    await _amain(
        [
            "--num-shards",
            "1",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--job-kwargs",
            '{"input_column_name": "text", "output_cols": "output"}',
        ]
    )

    # With --no-resume, we should recompute and therefore hit the raising transform_items.
    with pytest.raises(RuntimeError, match="should not be called"):
        await _amain(
            [
                "--num-shards",
                "1",
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--no-resume",
                "--job-kwargs",
                '{"input_column_name": "text", "output_cols": "output"}',
            ]
        )


@pytest.mark.asyncio
async def test_amain_no_checkpointing_forces_recompute(monkeypatch, tmp_path):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    checkpoint_dir = tmp_path / "checkpoints"

    df_in = pd.DataFrame({"text": ["hello", "world"]})
    df_in.to_parquet(input_path)

    monkeypatch.setenv("JOB_CLASS", "domyn_swarm.jobs.cli.run:DummySwarmJob")
    monkeypatch.setenv("MODEL", "mock-model")
    monkeypatch.setenv("ENDPOINT", "mock-endpoint")
    monkeypatch.setenv("INPUT_PARQUET", str(input_path))
    monkeypatch.setenv("OUTPUT_PARQUET", str(output_path))
    monkeypatch.setenv("JOB_KWARGS", '{"input_column_name": "text", "output_cols": "output"}')
    monkeypatch.setattr(run_mod, "_load_cls", lambda path: DummySwarmJob)

    # First run creates checkpoints
    await _amain(
        [
            "--num-shards",
            "1",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--job-kwargs",
            '{"input_column_name": "text", "output_cols": "output"}',
        ]
    )

    async def _raise_transform_items(self, items: list):
        raise RuntimeError("must be called when checkpointing is disabled")

    monkeypatch.setattr(DummySwarmJob, "transform_items", _raise_transform_items)
    with pytest.raises(RuntimeError, match="checkpointing is disabled"):
        await _amain(
            [
                "--num-shards",
                "1",
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--no-checkpointing",
                "--job-kwargs",
                '{"input_column_name": "text", "output_cols": "output"}',
            ]
        )


@pytest.mark.asyncio
async def test_amain_ray_backend_requires_ray_address(monkeypatch, tmp_path):
    """Fail early if a Ray job is started without a ray address."""
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"

    monkeypatch.delenv("DOMYN_SWARM_RAY_ADDRESS", raising=False)
    monkeypatch.delenv("RAY_ADDRESS", raising=False)

    class _RayBackend:
        name = "ray"

        def read(self, *args, **kwargs):
            raise AssertionError("backend.read should not be called when ray address is missing")

    monkeypatch.setattr(run_mod, "get_backend", lambda name: _RayBackend())
    monkeypatch.setattr(
        run_mod,
        "build_job_from_args",
        lambda args: (DummySwarmJob, {"data_backend": "ray"}),
    )

    args = run_mod.parse_args(
        [
            "--job-class",
            "domyn_swarm.jobs.cli.run:DummySwarmJob",
            "--model",
            "m",
            "--endpoint",
            "http://e",
            "--input-parquet",
            str(input_path),
            "--output-parquet",
            str(output_path),
            "--checkpoint-tag",
            "t",
        ]
    )

    with pytest.raises(ValueError, match="explicit ray address"):
        await run_mod._amain(args)


@pytest.mark.asyncio
async def test_amain_ray_backend_forwards_ray_address(monkeypatch, tmp_path):
    """Forward ray address into run_job_unified when provided."""
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"

    class _RayBackend:
        name = "ray"

        def read(self, *args, **kwargs):
            return "data"

        def write(self, *args, **kwargs):
            return None

    monkeypatch.setattr(run_mod, "get_backend", lambda name: _RayBackend())
    monkeypatch.setattr(
        run_mod,
        "build_job_from_args",
        lambda args: (DummySwarmJob, {"data_backend": "ray"}),
    )

    seen: dict[str, object] = {}

    async def _fake_run_job_unified(*args, **kwargs):
        seen.update(kwargs)
        return "result"

    monkeypatch.setattr(run_mod, "run_job_unified", _fake_run_job_unified)
    monkeypatch.setattr(run_mod, "_write_result", lambda *a, **k: None)

    args = run_mod.parse_args(
        [
            "--job-class",
            "domyn_swarm.jobs.cli.run:DummySwarmJob",
            "--model",
            "m",
            "--endpoint",
            "http://e",
            "--input-parquet",
            str(input_path),
            "--output-parquet",
            str(output_path),
            "--checkpoint-tag",
            "t",
            "--ray-address",
            "ray://head:10001",
        ]
    )

    await run_mod._amain(args)
    assert seen["ray_address"] == "ray://head:10001"


@pytest.mark.asyncio
async def test_amain_forwards_global_resume(monkeypatch, tmp_path):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    pd.DataFrame({"messages": [1, 2]}).to_parquet(input_path)

    monkeypatch.setattr(
        run_mod,
        "build_job_from_args",
        lambda args: (DummySwarmJob, {"data_backend": "pandas"}),
    )

    seen: dict[str, object] = {}

    async def _fake_run_job_unified(*args, **kwargs):
        seen.update(kwargs)
        return pd.DataFrame({"messages": [1, 2], "output": ["a", "b"]})

    monkeypatch.setattr(run_mod, "run_job_unified", _fake_run_job_unified)
    monkeypatch.setattr(run_mod, "_write_result", lambda *a, **k: None)

    args = run_mod.parse_args(
        [
            "--job-class",
            "domyn_swarm.jobs.cli.run:DummySwarmJob",
            "--model",
            "m",
            "--endpoint",
            "http://e",
            "--input-parquet",
            str(input_path),
            "--output-parquet",
            str(output_path),
            "--global-resume",
        ]
    )

    await run_mod._amain(args)
    assert seen["global_resume"] is True


def test_main_wrapper(monkeypatch):
    # Use parse_args via monkeypatch to avoid invoking asyncio in this test
    with patch("domyn_swarm.jobs.cli.run._amain") as mock_amain:
        from domyn_swarm.jobs.cli.run import main

        main(
            [
                "--job-class",
                "mod:Cls",
                "--model",
                "gpt",
                "--input-parquet",
                "in.pq",
                "--output-parquet",
                "out.pq",
                "--endpoint",
                "ep",
            ]
        )
        mock_amain.assert_called_once()


def test_nthreads_is_accepted_as_a_deprecated_alias_for_num_shards():
    """In-flight jobs launched by an older driver keep parsing."""
    args = parse_args(
        [
            "--job-class=pkg:Cls",
            "--model=m",
            "--input-parquet=in.parquet",
            "--output-parquet=out.parquet",
            "--endpoint=http://e",
            "--nthreads",
            "7",
        ]
    )
    assert args.num_shards == 7
