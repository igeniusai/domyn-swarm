import importlib

import numpy as np
import pandas as pd
import pytest


@pytest.mark.asyncio
async def test_run_sharded_single_shard(mocker):
    # Import the module that contains run_sharded
    mod = importlib.import_module("domyn_swarm.jobs.runner")

    # --- Arrange -------------------------------------------------------------
    df = pd.DataFrame(
        {"messages": [f"m{i}" for i in range(6)]},
        index=pd.RangeIndex(0, 6),
    )
    store_uri = "file:///tmp/out.parquet"

    # Capture constructed store URIs
    created_store_uris = []

    class FakeStore:
        def __init__(self, uri: str):
            created_store_uris.append(uri)
            self.uri = uri

    # Capture JobRunner construction and run calls
    created_runners = []
    run_calls = []

    class FakeJobRunner:
        def __init__(self, store, cfg):
            self.store = store
            self.cfg = cfg
            created_runners.append(self)

        async def run(self, job, sub_df, *, input_col, output_cols):
            # Record that we were called and with which indices
            run_calls.append(
                {
                    "idx": sub_df.index.tolist(),
                    "input_col": input_col,
                    "output_cols": output_cols,
                    "store_uri": getattr(self.store, "uri", None),
                }
            )
            # Produce a deterministic output: add a 'result' (or first output col) column
            out = sub_df.copy()
            col = (
                output_cols[0]
                if isinstance(output_cols, list) and output_cols
                else "result"
            )
            out[col] = [f"{x}-out" for x in sub_df[input_col].tolist()]
            return out

    # Patch in the module
    mocker.patch.object(mod, "ParquetShardStore", FakeStore)
    mocker.patch.object(mod, "JobRunner", FakeJobRunner)

    factory_calls = {"n": 0}

    def job_factory():
        factory_calls["n"] += 1
        return object()

    # --- Act ----------------------------------------------------------------
    result = await mod.run_sharded(
        job_factory,
        df,
        input_col="messages",
        output_cols=["result"],
        store_uri=store_uri,
        nshards=1,
        cfg=None,
    )

    # --- Assert --------------------------------------------------------------
    # Only one store constructed with the base URI (no suffix for nshards=1)
    assert created_store_uris == [store_uri]

    # Only one JobRunner and one run() call
    assert len(created_runners) == 1
    assert len(run_calls) == 1

    # Factory called exactly once
    assert factory_calls["n"] == 1

    # Output shape & content
    assert list(result.index) == list(df.index)
    assert "result" in result.columns
    assert result["result"].tolist() == [f"m{i}-out" for i in range(6)]

    # Correct args were forwarded to JobRunner.run
    call = run_calls[0]
    assert call["input_col"] == "messages"
    assert call["output_cols"] == ["result"]
    assert call["store_uri"] == store_uri


@pytest.mark.asyncio
async def test_run_sharded_multiple_shards(mocker):
    mod = importlib.import_module("domyn_swarm.jobs.runner")

    # --- Arrange -------------------------------------------------------------
    # 10 rows, split into 3 shards
    n = 10
    df = pd.DataFrame({"messages": [f"m{i}" for i in range(n)]}, index=np.arange(n))
    base_uri = "file:///tmp/out.parquet"

    created_store_uris = []

    class FakeStore:
        def __init__(self, uri: str):
            created_store_uris.append(uri)
            self.uri = uri

    created_runners = []
    run_calls = []

    class FakeJobRunner:
        def __init__(self, store, cfg):
            self.store = store
            self.cfg = cfg
            created_runners.append(self)

        async def run(self, job, sub_df, *, input_col, output_cols):
            run_calls.append(
                {
                    "idx": sub_df.index.tolist(),
                    "input_col": input_col,
                    "output_cols": output_cols,
                    "store_uri": getattr(self.store, "uri", None),
                }
            )
            out = sub_df.copy()
            col = (
                output_cols[0]
                if isinstance(output_cols, list) and output_cols
                else "result"
            )
            out[col] = [f"{x}-out" for x in sub_df[input_col].tolist()]
            return out

    mocker.patch.object(mod, "ParquetShardStore", FakeStore)
    mocker.patch.object(mod, "JobRunner", FakeJobRunner)

    factory_calls = {"n": 0}

    def job_factory():
        factory_calls["n"] += 1
        return object()

    nshards = 3

    # Derive expected shard URIs using the same logic as the function
    expected_uris = {
        base_uri.replace(".parquet", f"_shard{i}.parquet") for i in range(nshards)
    }

    # --- Act ----------------------------------------------------------------
    result = await mod.run_sharded(
        job_factory,
        df,
        input_col="messages",
        output_cols=["result"],
        store_uri=base_uri,
        nshards=nshards,
        cfg=None,
    )

    # --- Assert --------------------------------------------------------------
    # We expect exactly nshards store URIs, each with the shard suffix
    assert set(created_store_uris) == expected_uris
    assert len(created_store_uris) == nshards

    # One JobRunner per shard, one run() call per shard
    assert len(created_runners) == nshards
    assert len(run_calls) == nshards

    # Factory called once per shard
    assert factory_calls["n"] == nshards

    # Result is concatenation of shard outputs, sorted by index
    assert list(result.index) == list(range(n))
    assert "result" in result.columns
    assert result["result"].tolist() == [f"m{i}-out" for i in range(n)]

    # Check each run call got correct args
    for call in run_calls:
        assert call["input_col"] == "messages"
        assert call["output_cols"] == ["result"]
        assert call["store_uri"] in expected_uris


@pytest.mark.asyncio
async def test_run_sharded_multiple_shards_respects_splitting(mocker):
    """
    Extra check: indices passed to each shard runner are disjoint and cover the full set.
    """
    mod = importlib.import_module("domyn_swarm.jobs.runner")

    df = pd.DataFrame({"messages": [f"m{i}" for i in range(11)]}, index=np.arange(11))
    base_uri = "file:///tmp/out.parquet"

    class FakeStore:
        def __init__(self, uri: str):
            self.uri = uri

    run_indices = []

    class FakeJobRunner:
        def __init__(self, store, cfg):
            self.store = store
            self.cfg = cfg

        async def run(self, job, sub_df, *, input_col, output_cols):
            run_indices.append(tuple(sub_df.index.tolist()))
            out = sub_df.copy()
            col = (
                output_cols[0]
                if isinstance(output_cols, list) and output_cols
                else "result"
            )
            out[col] = [f"{x}-out" for x in sub_df[input_col].tolist()]
            return out

    mocker.patch.object(mod, "ParquetShardStore", FakeStore)
    mocker.patch.object(mod, "JobRunner", FakeJobRunner)

    def job_factory():
        return object()

    nshards = 4
    _ = await mod.run_sharded(
        job_factory,
        df,
        input_col="messages",
        output_cols=["result"],
        store_uri=base_uri,
        nshards=nshards,
        cfg=None,
    )

    # Collect and validate coverage
    seen = []
    for idxs in run_indices:
        seen.extend(list(idxs))

    # Disjoint coverage in any order, but overall set equals original indices
    assert set(seen) == set(df.index)
    # Sanity: no duplicate indices across shards
    assert len(seen) == len(set(seen))
