# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""The shared pipeline's control flow, exercised with a stub adapter."""

import pandas as pd
import pytest

from domyn_swarm.data.backends.registry import get_backend
from domyn_swarm.jobs.api.base import OutputJoinMode
from domyn_swarm.jobs.execution.frame_ops import PandasFrameOps, ShardSpec
from domyn_swarm.jobs.execution.pipeline import run_sharded_pipeline


def _spec(**overrides) -> ShardSpec:
    base = {
        "input_col": "messages",
        "output_cols": ["output"],
        "id_col": "doc_id",
        "checkpoint_every": 8,
        "checkpointing": True,
        "output_mode": OutputJoinMode.APPEND,
    }
    base.update(overrides)
    return ShardSpec(**base)


class RecordingOps(PandasFrameOps):
    """Pandas adapter that records how many shards the pipeline ran."""

    def __init__(self, backend):
        super().__init__(backend)
        self.shards_run = 0

    async def run_shard(self, job_factory, frame, store_uri, spec):
        self.shards_run += 1
        out = frame.copy(deep=False)
        out["output"] = [f"out-{v}" for v in out[spec.input_col]]
        return out


@pytest.mark.asyncio
async def test_pipeline_runs_one_shard_per_index_group(tmp_path) -> None:
    """The pipeline runs exactly `nshards` shards."""
    ops = RecordingOps(get_backend("pandas"))
    df = pd.DataFrame({"doc_id": [1, 2, 3, 4], "messages": [1, 2, 3, 4]})

    out = await run_sharded_pipeline(
        ops=ops,
        job_factory=lambda: None,
        data=df,
        spec=_spec(),
        require_id=True,
        nshards=2,
        shard_mode="id",
        global_resume=False,
        store_uri=f"file://{tmp_path / 'out.parquet'}",
    )

    assert ops.shards_run == 2
    assert sorted(out["doc_id"].tolist()) == [1, 2, 3, 4]


@pytest.mark.asyncio
async def test_pipeline_single_shard_skips_sharding(tmp_path) -> None:
    """With nshards <= 1 the pipeline runs the frame once, unsharded."""
    ops = RecordingOps(get_backend("pandas"))
    df = pd.DataFrame({"doc_id": [1, 2], "messages": [1, 2]})

    out = await run_sharded_pipeline(
        ops=ops,
        job_factory=lambda: None,
        data=df,
        spec=_spec(),
        require_id=True,
        nshards=1,
        shard_mode="id",
        global_resume=False,
        store_uri=f"file://{tmp_path / 'out.parquet'}",
    )

    assert ops.shards_run == 1
    assert out["output"].tolist() == ["out-1", "out-2"]


@pytest.mark.asyncio
async def test_pipeline_rejects_unknown_shard_mode(tmp_path) -> None:
    """An unsupported shard_mode fails before any shard runs."""
    ops = RecordingOps(get_backend("pandas"))
    df = pd.DataFrame({"doc_id": [1, 2], "messages": [1, 2]})

    with pytest.raises(ValueError, match="Unsupported shard_mode"):
        await run_sharded_pipeline(
            ops=ops,
            job_factory=lambda: None,
            data=df,
            spec=_spec(),
            require_id=True,
            nshards=2,
            shard_mode="nonsense",
            global_resume=False,
            store_uri=f"file://{tmp_path / 'out.parquet'}",
        )

    assert ops.shards_run == 0


@pytest.mark.asyncio
async def test_pipeline_requires_the_id_column_when_declared(tmp_path) -> None:
    """A declared id column that is absent from the input is an error."""
    ops = RecordingOps(get_backend("pandas"))
    df = pd.DataFrame({"messages": [1, 2]})

    with pytest.raises(ValueError, match="doc_id"):
        await run_sharded_pipeline(
            ops=ops,
            job_factory=lambda: None,
            data=df,
            spec=_spec(),
            require_id=True,
            nshards=1,
            shard_mode="id",
            global_resume=False,
            store_uri=f"file://{tmp_path / 'out.parquet'}",
        )
