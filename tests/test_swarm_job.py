import pytest
import pandas as pd
from domyn_swarm.jobs.base import SwarmJob


class DummyJob(SwarmJob):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []

    async def transform(self, df: pd.DataFrame):
        await self.batched(df[self.input_column_name].tolist(), self.fake_fn)

    async def fake_fn(self, x):
        self.calls.append(x)
        return f"out_{x}"


@pytest.mark.asyncio
async def test_swarm_job_checkpointing(tmp_path):
    df = pd.DataFrame({"messages": ["hi", "yo", "hello"]})
    job = DummyJob(
        endpoint="http://localhost", model="fake", checkpoint_dir=str(tmp_path)
    )

    result = await job.run(df, tag="test", checkpoint_dir=tmp_path)
    assert set(result.columns) >= {"messages", "result"}
    assert job.calls == ["hi", "yo", "hello"]
