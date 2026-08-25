# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from domyn_swarm.jobs.api.base import SwarmJob


class DummyJob(SwarmJob):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []

    async def transform_items(self, items):
        return [await self.fake_fn(item) for item in items]

    async def fake_fn(self, x):
        self.calls.append(x)
        return f"out_{x}"


@pytest.mark.asyncio
async def test_swarm_job_checkpointing(tmp_path):
    df = pd.DataFrame({"messages": ["hi", "yo", "hello"]})
    job = DummyJob(endpoint="http://localhost", model="fake", checkpoint_dir=str(tmp_path))

    result = await job.run(df, tag="test", checkpoint_dir=tmp_path)
    assert set(result.columns) >= {"messages", "result"}
    assert job.calls == ["hi", "yo", "hello"]
