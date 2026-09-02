# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for `_run_pandas` that aren't part of the behaviour contract.

`tests/jobs/execution/test_engine_characterization.py` pins cross-engine
agreement and must not be edited to make a refactor pass. This file is for
bugs found *during* the consolidation refactor that the characterization
suite doesn't happen to exercise (e.g. because every one of its polars-backend
tests routes through `runner="arrow"`, not the default `runner="pandas"`).
"""

import pandas as pd
import pytest

from domyn_swarm.jobs import SwarmJob
from domyn_swarm.jobs.api.base import OutputJoinMode
from domyn_swarm.jobs.cli.run import run_job_unified


class _EchoJob(SwarmJob):
    """Deterministic job: output depends only on the input item."""

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
async def test_pandas_runner_polars_backend_global_resume_returns_polars_frame(tmp_path):
    """The pandas runner's global-resume path must convert to the target backend once.

    `data_backend="polars"` with the default `runner="pandas"` (not "arrow", which
    routes to a different function entirely) and `nshards>1` with
    `global_resume=True` and `checkpointing=True` exercises `_run_pandas`'s
    global-resume branch: `run_sharded_pipeline`'s `finalize` hook
    (`_finalize_global_resume`) already converts pandas -> polars before
    returning, and `_run_pandas`'s own tail line converts again. Converting an
    already-polars frame with `PolarsBackend.from_pandas` raises `TypeError`.
    """
    pytest.importorskip("polars")
    import polars as pl

    df = pd.DataFrame({"doc_id": [10, 11, 12, 13, 14], "messages": [1, 2, 3, 4, 5]})

    result = await run_job_unified(
        lambda: _EchoJob(id_column_name="doc_id"),
        df,
        input_col="messages",
        output_cols=["output"],
        store_uri=f"file://{tmp_path / 'out.parquet'}",
        nshards=2,
        shard_mode="id",
        global_resume=True,
        checkpointing=True,
        data_backend="polars",
        # runner defaults to "pandas".
    )

    assert isinstance(result, pl.DataFrame)
    got = sorted(
        zip(result["doc_id"].to_list(), result["output"].to_list(), strict=True),
        key=lambda row: row[0],
    )
    assert got == [
        (10, "out-1"),
        (11, "out-2"),
        (12, "out-3"),
        (13, "out-4"),
        (14, "out-5"),
    ]
