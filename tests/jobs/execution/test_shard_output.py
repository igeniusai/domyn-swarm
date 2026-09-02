# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""`shard_output` controls sharded directory output on the pandas engine.

Before this change the pandas engine wrote one parquet file per shard
whenever `output_path` was a directory, ignoring the documented flag. The
flag now governs it.
"""

import pandas as pd
import pytest

from domyn_swarm.jobs import SwarmJob
from domyn_swarm.jobs.api.base import OutputJoinMode
from domyn_swarm.jobs.cli.run import run_job_unified


class ShardOutputJob(SwarmJob):
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

    async def transform_items(self, items: list):
        return [f"out-{i}" for i in items]


@pytest.mark.asyncio
async def test_shard_output_true_writes_one_file_per_shard(tmp_path):
    """Directory output still works when the flag is explicitly enabled.

    This behaviour predates the change (the pandas engine always wrote
    shard files for a directory `output_path`, flag or no flag) and is not
    itself new. This is a non-regression guard: it pins that gating
    directory output on `shard_output` does not also kill directory output
    for callers who legitimately want it, i.e. that the fix does not
    over-correct. It is expected to be green both before and after the
    change; see `test_shard_output_false_does_not_write_shard_files` for
    the test that actually proves the behaviour change.
    """
    out_dir = tmp_path / "outdir"
    out_dir.mkdir()
    df = pd.DataFrame({"doc_id": [1, 2, 3, 4], "messages": [1, 2, 3, 4]})

    await run_job_unified(
        lambda: ShardOutputJob(id_column_name="doc_id"),
        df,
        input_col="messages",
        output_cols=["output"],
        store_uri=f"file://{tmp_path / 'ckpt.parquet'}",
        nshards=2,
        checkpointing=True,
        output_path=out_dir,
        shard_output=True,
        runner="pandas",
    )

    assert len(list(out_dir.glob("*.parquet"))) == 2


@pytest.mark.asyncio
async def test_shard_output_false_does_not_write_shard_files(tmp_path):
    """Without the flag, a directory output path no longer writes shard files.

    This is a deliberate behaviour change: the pandas engine used to do this
    implicitly, ignoring the flag.
    """
    out_dir = tmp_path / "outdir"
    out_dir.mkdir()
    df = pd.DataFrame({"doc_id": [1, 2, 3, 4], "messages": [1, 2, 3, 4]})

    result = await run_job_unified(
        lambda: ShardOutputJob(id_column_name="doc_id"),
        df,
        input_col="messages",
        output_cols=["output"],
        store_uri=f"file://{tmp_path / 'ckpt.parquet'}",
        nshards=2,
        checkpointing=True,
        output_path=out_dir,
        shard_output=False,
        runner="pandas",
    )

    assert list(out_dir.glob("*.parquet")) == []
    assert sorted(result["doc_id"].tolist()) == [1, 2, 3, 4]


@pytest.mark.asyncio
async def test_shard_output_true_with_global_resume_keeps_completed_ids(tmp_path):
    """Resuming a partially-complete job under `shard_output=True` should not lose ids.

    Mirrors the partial-resume technique in
    `test_engine_characterization.test_checkpoints_are_portable_between_engines`:
    run a first pass so some ids are recorded in the per-shard checkpoint
    stores, then run again with `global_resume=True`. Here both passes also
    use `shard_output=True` against the same output directory, since that is
    how a user would actually invoke `--shard-output` across a resumed run.

    The already-done ids (1, 2) must not be reprocessed -- enforced by
    `ExplodingResumeJob` raising if it ever sees them again -- and the output
    directory should end up containing every id (1, 2, 3, 4) with the correct
    output. The second assertion currently fails: see the `xfail` reason.
    """
    store_uri = f"file://{tmp_path / 'ckpt.parquet'}"
    out_dir = tmp_path / "outdir"
    out_dir.mkdir()

    # First pass: only ids 1 and 2 exist yet, establishing partial per-shard
    # checkpoint state (and partial shard output files) to resume from.
    df_partial = pd.DataFrame({"doc_id": [1, 2], "messages": [1, 2]})
    await run_job_unified(
        lambda: ShardOutputJob(id_column_name="doc_id"),
        df_partial,
        input_col="messages",
        output_cols=["output"],
        store_uri=store_uri,
        nshards=2,
        checkpointing=True,
        global_resume=True,
        output_path=out_dir,
        shard_output=True,
        runner="pandas",
    )

    class ExplodingResumeJob(ShardOutputJob):
        async def transform_items(self, items: list):
            for item in items:
                if item in (1, 2):
                    raise AssertionError(f"re-processed already-done item: {item}")
            return [f"out-{i}" for i in items]

    # Second pass: the full dataset (1, 2, 3, 4), resuming via global_resume.
    df_full = pd.DataFrame({"doc_id": [1, 2, 3, 4], "messages": [1, 2, 3, 4]})
    await run_job_unified(
        lambda: ExplodingResumeJob(id_column_name="doc_id"),
        df_full,
        input_col="messages",
        output_cols=["output"],
        store_uri=store_uri,
        nshards=2,
        checkpointing=True,
        global_resume=True,
        output_path=out_dir,
        shard_output=True,
        runner="pandas",
    )

    written = pd.concat([pd.read_parquet(f) for f in sorted(out_dir.glob("*.parquet"))])
    assert sorted(written["doc_id"].tolist()) == [1, 2, 3, 4]


@pytest.mark.asyncio
async def test_shard_output_true_missing_id_column_raises(tmp_path):
    """A directory run with `shard_output=True` requires the declared id column.

    Reaches `_run_pandas_to_directory`'s own `require_id` check (the
    directory path validates independently of `_run_pandas`, since it never
    calls it).
    """
    out_dir = tmp_path / "outdir"
    out_dir.mkdir()
    df = pd.DataFrame({"other_id": [1, 2, 3, 4], "messages": [1, 2, 3, 4]})

    with pytest.raises(ValueError, match="doc_id"):
        await run_job_unified(
            lambda: ShardOutputJob(id_column_name="doc_id"),
            df,
            input_col="messages",
            output_cols=["output"],
            store_uri=f"file://{tmp_path / 'ckpt.parquet'}",
            nshards=2,
            checkpointing=True,
            output_path=out_dir,
            shard_output=True,
            runner="pandas",
        )


@pytest.mark.asyncio
async def test_shard_output_true_unsupported_shard_mode_raises(tmp_path):
    """An unrecognized `shard_mode` is rejected on the `shard_output=True` directory path."""
    out_dir = tmp_path / "outdir"
    out_dir.mkdir()
    df = pd.DataFrame({"doc_id": [1, 2, 3, 4], "messages": [1, 2, 3, 4]})

    with pytest.raises(ValueError, match="Unsupported shard_mode"):
        await run_job_unified(
            lambda: ShardOutputJob(id_column_name="doc_id"),
            df,
            input_col="messages",
            output_cols=["output"],
            store_uri=f"file://{tmp_path / 'ckpt.parquet'}",
            nshards=2,
            shard_mode="bogus",
            checkpointing=True,
            output_path=out_dir,
            shard_output=True,
            runner="pandas",
        )


@pytest.mark.asyncio
async def test_shard_output_true_missing_store_uri_raises(tmp_path):
    """A missing `store_uri` is rejected on the `shard_output=True` directory path.

    `checkpointing=True` alone is not enough: the directory path also needs a
    concrete store URI to build per-shard checkpoint stores from.
    """
    out_dir = tmp_path / "outdir"
    out_dir.mkdir()
    df = pd.DataFrame({"doc_id": [1, 2, 3, 4], "messages": [1, 2, 3, 4]})

    with pytest.raises(ValueError, match="store_uri"):
        await run_job_unified(
            lambda: ShardOutputJob(id_column_name="doc_id"),
            df,
            input_col="messages",
            output_cols=["output"],
            store_uri=None,
            nshards=2,
            checkpointing=True,
            output_path=out_dir,
            shard_output=True,
            runner="pandas",
        )


@pytest.mark.asyncio
async def test_polars_shard_output_true_with_global_resume_keeps_completed_ids(tmp_path):
    """The polars engine must not drop completed ids from its shard output either.

    The same defect as
    `test_shard_output_true_with_global_resume_keeps_completed_ids`, in a
    quieter shape: `PolarsJobRunner._stream_output_to_path` left-joins a
    shard's checkpoint outputs onto that shard's slice of the input, so a row
    filtered out by global resume was absent from the join's left side and
    therefore missing from the file that overwrote the previous run's. The
    directory was left looking complete while silently holding only the rows
    processed on the last run.

    Note the shape this needs: re-running the *identical* full frame does not
    reproduce it, because filtering out every id sends `_run_polars_sharded`
    down its zero-row early return, which never touches the directory. A
    partial first pass followed by a wider second pass is required.
    """
    pl = pytest.importorskip("polars")

    store_uri = f"file://{tmp_path / 'ckpt.parquet'}"
    out_dir = tmp_path / "outdir"
    out_dir.mkdir()
    common = {
        "input_col": "messages",
        "output_cols": ["output"],
        "store_uri": store_uri,
        "nshards": 2,
        "shard_mode": "id",
        "checkpointing": True,
        "global_resume": True,
        "output_path": out_dir,
        "shard_output": True,
        "engine": "polars",
        "data_backend": "polars",
    }

    # First pass: only ids 1 and 2 exist, establishing partial per-shard
    # checkpoint state and partial shard output files to resume from.
    await run_job_unified(
        lambda: ShardOutputJob(id_column_name="doc_id"),
        pl.DataFrame({"doc_id": [1, 2], "messages": [1, 2]}),
        **common,
    )

    class ExplodingResumeJob(ShardOutputJob):
        async def transform_items(self, items: list):
            for item in items:
                if item in (1, 2):
                    raise AssertionError(f"re-processed already-done item: {item}")
            return [f"out-{i}" for i in items]

    # Second pass: the full dataset. Ids 1 and 2 must not be reprocessed, and
    # must still be present in the output directory afterwards.
    await run_job_unified(
        lambda: ExplodingResumeJob(id_column_name="doc_id"),
        pl.DataFrame({"doc_id": [1, 2, 3, 4], "messages": [1, 2, 3, 4]}),
        **common,
    )

    parts = [pl.read_parquet(f) for f in sorted(out_dir.glob("*.parquet"))]
    written = pl.concat([p for p in parts if p.height], how="diagonal")
    assert sorted(written["doc_id"].to_list()) == [1, 2, 3, 4]
    assert sorted(written["output"].to_list()) == ["out-1", "out-2", "out-3", "out-4"]
