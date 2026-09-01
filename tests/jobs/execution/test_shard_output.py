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
