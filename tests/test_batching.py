import asyncio
import pytest
from domyn_swarm.jobs.batching import BatchExecutor


@pytest.mark.asyncio
async def test_batch_executor_runs_sequentially():
    items = list(range(10))

    async def echo(x):
        await asyncio.sleep(0.01)
        return x * 2

    executor = BatchExecutor(parallel=2, batch_size=5, retries=2)
    result = await executor.run(items, echo)
    assert result == [x * 2 for x in items]


@pytest.mark.asyncio
async def test_batch_executor_callback_invoked():
    items = list(range(6))
    flushed = []

    async def fn(x):
        return x + 1

    async def flush(out, ids):
        flushed.extend(ids)

    executor = BatchExecutor(parallel=3, batch_size=3, retries=1)
    await executor.run(items, fn, on_batch_done=flush)

    assert set(flushed) == set(range(6))