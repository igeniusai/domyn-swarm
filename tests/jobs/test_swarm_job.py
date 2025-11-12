# Copyright 2025 iGenius S.p.A
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import pytest

from domyn_swarm.jobs.base import SwarmJob


class DummyJob(SwarmJob):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []

    async def transform(self, df: pd.DataFrame):
        await self.batched(df[self.input_column_name].tolist(), self.fake_fn)

    async def transform_items(self, items):
        return await super().transform_items(items)

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
