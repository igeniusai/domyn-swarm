import asyncio
import logging

from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm

logger = logging.getLogger(__name__)


class BatchExecutor:
    """
    Executor for running tasks in parallel batches.
    This class handles batching of items and parallel execution of a function on those items.
    It supports retry logic and can execute callbacks after each batch.
    """

    def __init__(self, parallel: int, batch_size: int, retries: int):
        self.parallel = parallel
        self.batch_size = batch_size
        self.retries = retries

    async def run(self, items, fn, *, on_batch_done=None):
        """
        Run a function `fn` on `items` in parallel batches.

        Args:
            items: List of items to process.
            fn: Function to apply to each item or batch of items.
            on_batch_done: Optional callback to run after each batch is processed.
        """
        out = [None] * len(items)
        sem = asyncio.Semaphore(self.parallel)
        queue = asyncio.Queue()
        for idx, item in enumerate(items):
            queue.put_nowait((idx, item))

        lock = asyncio.Lock()
        completed = 0
        pending_ids = []

        fn = retry(
            wait=wait_exponential(multiplier=1, min=4, max=10),
            stop=stop_after_attempt(self.retries),
            reraise=True,
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )(fn)

        pbar = tqdm(total=min(self.batch_size, len(items)), leave=True)

        async def worker():
            nonlocal completed, pending_ids
            while not queue.empty():
                idx, item = await queue.get()
                async with sem:
                    out[idx] = (
                        await fn(*item) if isinstance(item, tuple) else await fn(item)
                    )

                async with lock:
                    completed += 1
                    pending_ids.append(idx)

                    flush_now = completed % self.batch_size == 0 or completed == len(
                        items
                    )
                    if flush_now and on_batch_done:
                        await on_batch_done(out, pending_ids)
                        pending_ids = []
                        pbar.reset(total=min(queue.qsize(), self.batch_size))

                    pbar.update(1)

        await asyncio.gather(*(worker() for _ in range(self.parallel)))
        pbar.close()
        return out
