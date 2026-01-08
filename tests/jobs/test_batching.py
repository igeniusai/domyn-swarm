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

import asyncio

import pytest

from domyn_swarm.jobs.batching import BatchExecutor


@pytest.mark.asyncio
async def test_batch_executor_runs_sequentially():
    items = list(range(10))

    async def echo(x):
        await asyncio.sleep(0.01)
        return x * 2

    executor = BatchExecutor(max_concurrency=2, checkpoint_interval=5, retries=2)
    result = await executor.run(items, echo, progress=False)
    assert result == [x * 2 for x in items]


@pytest.mark.asyncio
async def test_batch_executor_callback_invoked():
    items = list(range(6))
    flushed = []

    async def fn(x):
        return x + 1

    async def flush(out, ids):
        flushed.extend(ids)

    executor = BatchExecutor(max_concurrency=3, checkpoint_interval=3, retries=1)
    await executor.run(items, fn, on_batch_done=flush, progress=False)

    assert set(flushed) == set(range(6))


@pytest.mark.asyncio
async def test_batch_executor_flushes_tail_batch():
    items = list(range(5))
    flushed = []

    async def fn(x):
        return x * 10

    async def flush(out, ids):
        flushed.append(list(ids))

    executor = BatchExecutor(max_concurrency=2, checkpoint_interval=4, retries=1)
    result = await executor.run(items, fn, on_batch_done=flush, progress=False)

    assert result == [x * 10 for x in items]
    assert sorted({i for batch in flushed for i in batch}) == items
    assert len(flushed) == 2


@pytest.mark.asyncio
async def test_batch_executor_retries():
    items = [0, 1, 2]
    attempts = {}

    async def fn(x):
        attempts[x] = attempts.get(x, 0) + 1
        if x == 1 and attempts[x] < 2:
            raise RuntimeError("transient")
        return x + 5

    executor = BatchExecutor(max_concurrency=2, checkpoint_interval=2, retries=2)
    result = await executor.run(items, fn, progress=False)

    assert result == [x + 5 for x in items]
    assert attempts[1] == 2
