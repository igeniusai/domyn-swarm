import asyncio

import pandas as pd

from domyn_swarm.checkpoint.store import InMemoryStore
from domyn_swarm.jobs.api.base import OutputJoinMode, SwarmJob
from domyn_swarm.jobs.api.runner import JobRunner, RunnerConfig, normalize_batch_outputs


class DummyJob(SwarmJob):
    """Minimal job that echoes inputs for testing."""

    async def transform_items(self, items: list[str]) -> list[str]:
        """Echo items back for testing.

        Args:
            items: Input items to echo.

        Returns:
            The same items unchanged.
        """
        return items

    async def transform_streaming(self, items, *, on_flush, checkpoint_every):
        await on_flush(list(range(len(items))), items)


def test_normalize_batch_outputs_scalar_with_cols():
    """Normalizes scalar outputs when output columns are specified."""
    rows, cols = normalize_batch_outputs([1, 2], ["out"])
    assert rows == [1, 2]
    assert cols == ["out"]


def test_normalize_batch_outputs_dict_without_cols():
    """Allows dict outputs when columns are not specified."""
    rows, cols = normalize_batch_outputs([{"a": 1}], None)
    assert rows == [{"a": 1}]
    assert cols is None


def test_job_runner_appends_outputs():
    """Appends outputs to the input DataFrame."""
    df = pd.DataFrame({"text": ["a", "b"]})
    runner = JobRunner(InMemoryStore(), RunnerConfig(checkpoint_every=1))
    out = asyncio.run(
        runner.run(
            DummyJob(endpoint="http://localhost", model="m"),
            df,
            input_col="text",
            output_cols=["out"],
            output_mode="append",
        )
    )
    assert "out" in out.columns
    assert out["out"].tolist() == ["a", "b"]


def test_job_runner_io_only():
    """Returns id/input/output columns when IO_ONLY is requested."""
    df = pd.DataFrame({"text": ["a", "b"]})
    runner = JobRunner(InMemoryStore(), RunnerConfig(checkpoint_every=1))
    out = asyncio.run(
        runner.run(
            DummyJob(endpoint="http://localhost", model="m"),
            df,
            input_col="text",
            output_cols=["out"],
            output_mode=OutputJoinMode.IO_ONLY,
        )
    )
    assert set(out.columns) == {"_row_id", "text", "out"}
