import pandas as pd
import pytest

import domyn_swarm.checkpoint.store as store_mod
from domyn_swarm.checkpoint.store import FlushBatch, ParquetShardStore
from domyn_swarm.jobs.base import OutputJoinMode, SwarmJob
import domyn_swarm.jobs.runner as runner_mod
from domyn_swarm.jobs.runner import JobRunner, RunnerConfig


@pytest.mark.asyncio
async def test_parquet_shard_store_flush_uses_to_thread(monkeypatch, tmp_path) -> None:
    """Ensure checkpoint flushing is offloaded so it can't block the event loop."""
    called = {"count": 0}

    async def fake_to_thread(fn, *args, **kwargs):
        called["count"] += 1
        return fn(*args, **kwargs)

    monkeypatch.setattr(store_mod.asyncio, "to_thread", fake_to_thread)

    store = ParquetShardStore(f"file://{tmp_path}/out.parquet")
    await store.flush(FlushBatch(ids=[1], rows=[{"a": 1}]), output_cols=None)

    assert called["count"] == 1


@pytest.mark.asyncio
async def test_job_runner_finalize_uses_to_thread(monkeypatch) -> None:
    """Ensure shard finalization (merge/write) doesn't stall other concurrent shards."""
    called = {"count": 0}

    async def fake_to_thread(fn, *args, **kwargs):
        called["count"] += 1
        return fn(*args, **kwargs)

    monkeypatch.setattr(runner_mod.asyncio, "to_thread", fake_to_thread)

    class DummyStore:
        """A minimal store for exercising JobRunner.finalize offloading."""

        def __init__(self) -> None:
            self.id_col = "_row_id"

        def prepare(self, df: pd.DataFrame, id_col: str) -> pd.DataFrame:
            self.id_col = id_col
            return df

        async def flush(self, batch: FlushBatch, output_cols: list[str] | None) -> None:
            return None

        def finalize(self) -> pd.DataFrame:
            out = pd.DataFrame({self.id_col: [0, 1], "output": ["x", "y"]})
            return out.set_index(self.id_col, drop=True)

    class DummyJob(SwarmJob):
        """A job that transforms items locally (no network)."""

        api_version = 2

        async def transform_items(self, items: list[object]) -> list[str]:
            return ["ok" for _ in items]

    df = pd.DataFrame({"messages": ["a", "b"]})
    job = DummyJob(
        endpoint="http://dummy",
        model="dummy",
        input_column_name="messages",
        output_cols="output",
        max_concurrency=1,
        retries=1,
        output_mode=OutputJoinMode.APPEND,
    )
    runner = JobRunner(DummyStore(), RunnerConfig(checkpoint_every=1))
    out = await runner.run(job, df, input_col="messages", output_cols=["output"])

    assert called["count"] == 1
    assert out["output"].tolist() == ["x", "y"]
