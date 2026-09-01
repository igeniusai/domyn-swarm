# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""`engine` selects the execution path; the old pair still maps onto it."""

import pandas as pd
import pytest

from domyn_swarm.jobs import SwarmJob
from domyn_swarm.jobs.api.base import OutputJoinMode
from domyn_swarm.jobs.execution.dispatch import resolve_engine, run_job_unified


@pytest.mark.parametrize(
    "data_backend,runner,expected",
    [
        ("ray", "pandas", "ray"),
        ("ray", "arrow", "ray"),
        ("polars", "arrow", "polars"),
        ("polars", "pandas", "pandas"),
        ("pandas", "arrow", "arrow"),
        ("pandas", "pandas", "pandas"),
    ],
)
def test_legacy_pair_maps_onto_engine(data_backend, runner, expected):
    """The deprecated pair keeps its exact current routing, quirks included."""
    assert resolve_engine(engine=None, data_backend=data_backend, runner=runner) == expected


def test_engine_takes_precedence_over_the_legacy_pair():
    """An explicit engine wins."""
    assert resolve_engine(engine="arrow", data_backend="polars", runner="pandas") == "arrow"


def test_unknown_engine_is_rejected():
    """An unsupported engine name fails with the valid choices named."""
    with pytest.raises(ValueError, match="pandas"):
        resolve_engine(engine="duckdb", data_backend=None, runner="pandas")


def test_legacy_runner_warns():
    """Passing the deprecated runner parameter warns once."""
    with pytest.warns(DeprecationWarning, match="runner"):
        resolve_engine(engine=None, data_backend="pandas", runner="arrow", warn=True)


class _SelectionJob(SwarmJob):
    """Minimal streaming job used to exercise `run_job_unified` end to end."""

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
async def test_engine_ray_rejects_a_non_ray_backend(tmp_path):
    """An explicit `engine="ray"` with an unresolved-ray backend fails up front."""
    df = pd.DataFrame({"messages": [1, 2]})
    with pytest.raises(ValueError, match=r"ray.*pandas|pandas.*ray"):
        await run_job_unified(
            _SelectionJob,
            df,
            input_col="messages",
            output_cols=["output"],
            store_uri=f"file://{tmp_path / 'out.parquet'}",
            engine="ray",
        )


@pytest.mark.asyncio
async def test_engine_polars_rejects_a_non_polars_backend(tmp_path):
    """An explicit `engine="polars"` with a pandas backend fails up front."""
    df = pd.DataFrame({"messages": [1, 2]})
    with pytest.raises(ValueError, match="polars"):
        await run_job_unified(
            _SelectionJob,
            df,
            input_col="messages",
            output_cols=["output"],
            store_uri=f"file://{tmp_path / 'out.parquet'}",
            engine="polars",
        )


@pytest.mark.asyncio
async def test_engine_arrow_runs_on_a_polars_backend(tmp_path):
    """`engine="arrow"` on a polars backend is a valid, previously-unreachable combo.

    The old `runner`/`data_backend` pair forced `runner="arrow"` with
    `data_backend="polars"` into the polars engine (`_run_polars`); naming the
    engine directly must actually run the Arrow engine and return real rows,
    proving the new guard does not also reject this legitimate combination.
    """
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"messages": [1, 2, 3]})

    out_df = await run_job_unified(
        _SelectionJob,
        df,
        input_col="messages",
        output_cols=["output"],
        store_uri=f"file://{tmp_path / 'out.parquet'}",
        data_backend="polars",
        engine="arrow",
    )

    assert isinstance(out_df, pl.DataFrame)
    assert sorted(out_df["output"].to_list()) == ["out-1", "out-2", "out-3"]
