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
import logging
import threading

from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm

logger = logging.getLogger(__name__)


class BatchExecutor:
    """
    Executor for running tasks in parallel batches with retry logic and progress tracking.

    This class provides a robust framework for processing large collections of items
    in parallel with configurable concurrency limits, automatic retries, and batch
    checkpointing capabilities.

    Key Features:
    - Parallel execution with configurable concurrency limits
    - Exponential backoff retry logic for failed tasks
    - Batch processing with configurable checkpoint intervals
    - Progress tracking with visual progress bars
    - Optional callback execution after each batch completion

    Args:
        max_concurrency (int): Maximum number of concurrent tasks to run
        checkpoint_interval (int): Number of items to process before triggering batch callback
        retries (int): Maximum number of retry attempts for failed tasks

    Example:
        >>> executor = BatchExecutor(max_concurrency=10, checkpoint_interval=100, retries=3)
        >>> results = await executor.run(items, process_item, on_batch_done=save_checkpoint)
    """

    def __init__(self, max_concurrency: int, checkpoint_interval: int, retries: int):
        self.max_concurrency = max_concurrency
        self.checkpoint_interval = checkpoint_interval
        self.retries = retries

    async def run(self, items, fn, *, on_batch_done=None, progress: bool = True):
        """
        Run a function `fn` on `items` in parallel batches.

        Args:
            items: List of items to process.
            fn: Function to apply to each item or batch of items.
            on_batch_done: Optional callback to run after each batch is processed.
            progress: Whether to display progress bars.
        """
        out = [None] * len(items)
        sem = asyncio.Semaphore(self.max_concurrency)
        queue: asyncio.Queue[tuple[int, object]] = asyncio.Queue()
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

        thread_name = threading.current_thread().name

        if progress:
            total_progress_bar = tqdm(
                total=len(items),
                desc=f"[{thread_name}] Processing all items in worker",
                leave=True,
                unit="sample",
            )
            pbar = tqdm(
                total=min(self.checkpoint_interval, len(items)),
                leave=True,
                desc=f"[{thread_name}] Processing batch",
                unit="sample",
            )
        else:
            total_progress_bar = None
            pbar = None

        async def worker():
            nonlocal completed, pending_ids
            while True:
                try:
                    idx, item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                async with sem:
                    out[idx] = await fn(*item) if isinstance(item, tuple) else await fn(item)
                queue.task_done()

                async with lock:
                    completed += 1
                    pending_ids.append(idx)

                    flush_now = completed % self.checkpoint_interval == 0 or completed == len(items)
                    if flush_now and on_batch_done:
                        await on_batch_done(out, pending_ids)
                        if total_progress_bar is not None:
                            total_progress_bar.update(len(pending_ids))
                        pending_ids = []
                        if pbar is not None:
                            pbar.reset(total=min(queue.qsize(), self.checkpoint_interval))

                    if pbar is not None:
                        pbar.update(1)

        await asyncio.gather(*(worker() for _ in range(self.max_concurrency)))
        if pending_ids and on_batch_done:
            await on_batch_done(out, pending_ids)
            if total_progress_bar is not None:
                total_progress_bar.update(len(pending_ids))
        if pbar is not None:
            pbar.close()
        if total_progress_bar is not None:
            total_progress_bar.close()
        return out
