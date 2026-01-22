import pytest

from domyn_swarm.jobs.batching import BatchExecutor


@pytest.mark.asyncio
async def test_batch_executor_run_streaming_returns_all_outputs():
    items = list(range(10))
    executor = BatchExecutor(max_concurrency=3, checkpoint_interval=4, retries=1)

    seen: dict[int, int] = {}
    batch_sizes: list[int] = []

    async def fn(x: int) -> int:
        return x * 2

    async def on_batch_done(outputs: list[int], idxs: list[int]) -> None:
        batch_sizes.append(len(idxs))
        seen.update({i: out for i, out in zip(idxs, outputs, strict=True)})

    await executor.run_streaming(items, fn, on_batch_done=on_batch_done, progress=False)

    assert seen == {i: i * 2 for i in range(10)}
    assert sorted(batch_sizes) == [2, 4, 4]
