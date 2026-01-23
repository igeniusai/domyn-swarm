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
from collections.abc import Awaitable, Callable, Sequence
import logging
import threading
from typing import Any

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

    async def run(
        self,
        items: Sequence[Any],
        fn: Callable[..., Awaitable[Any]],
        *,
        on_batch_done=None,
        progress: bool = True,
        on_progress=None,
        on_batch_progress=None,
    ):
        """
        Run a function `fn` on `items` in parallel batches.

        Args:
            items: List of items to process.
            fn: Function to apply to each item or batch of items.
            on_batch_done: Optional callback to run after each batch is processed.
            progress: Whether to display progress bars.
            on_progress: Optional callback with (completed, total) updates.
            on_batch_progress: Optional callback with (batch_completed, batch_total) updates.

        Example:
            >>> async def on_progress(done, total):
            ...     print(f"total progress {done}/{total}")
            >>> async def on_batch_progress(done, total):
            ...     print(f"batch progress {done}/{total}")
            >>> executor = BatchExecutor(max_concurrency=4, checkpoint_interval=8, retries=2)
            >>> results = await executor.run(
            ...     items,
            ...     fn,
            ...     progress=False,
            ...     on_progress=on_progress,
            ...     on_batch_progress=on_batch_progress,
            ... )
        """
        out = [None] * len(items)
        sem = asyncio.Semaphore(self.max_concurrency)
        queue = self._init_queue(items)
        lock = asyncio.Lock()
        completed = 0
        pending_ids: list[int] = []

        fn = self._wrap_retry(fn)
        total_progress_bar, pbar = self._init_progress(progress, len(items))

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
                    completed, pending_ids = await self._after_item(
                        completed=completed,
                        pending_ids=pending_ids,
                        idx=idx,
                        out=out,
                        on_batch_done=on_batch_done,
                        on_progress=on_progress,
                        on_batch_progress=on_batch_progress,
                        total=len(items),
                        queue_size=queue.qsize(),
                        total_progress_bar=total_progress_bar,
                        pbar=pbar,
                    )

        await asyncio.gather(*(worker() for _ in range(self.max_concurrency)))
        if pending_ids and on_batch_done:
            await on_batch_done(out, pending_ids)
            if total_progress_bar is not None:
                total_progress_bar.update(len(pending_ids))
        self._finalize_progress(total_progress_bar, pbar)
        return out

    async def run_streaming(
        self,
        items: Sequence[Any],
        fn: Callable[..., Awaitable[Any]],
        *,
        on_batch_done: Callable[[list[Any], list[int]], Awaitable[None]] | None = None,
        progress: bool = True,
        on_progress: Callable[[int, int], Awaitable[None]] | None = None,
        on_batch_progress: Callable[[int, int], Awaitable[None]] | None = None,
    ) -> None:
        """Run a batched async pipeline without retaining all outputs.

        Args:
            items: Sequence of items to process.
            fn: Coroutine function applied to each item.
            on_batch_done: Optional callback invoked as `await on_batch_done(outputs, idxs)`.
            progress: Whether to display progress bars.
            on_progress: Optional callback with `(completed, total)` updates.
            on_batch_progress: Optional callback with `(batch_completed, batch_total)` updates.
        """
        total = len(items)
        if total == 0:
            return

        sem = asyncio.Semaphore(self.max_concurrency)
        idx_lock = asyncio.Lock()
        state_lock = asyncio.Lock()
        next_idx = 0
        completed = 0
        pending_ids: list[int] = []
        pending_out: dict[int, Any] = {}

        fn = self._wrap_retry(fn)
        total_progress_bar, pbar = self._init_progress(progress, total)

        async def worker():
            nonlocal completed, pending_ids, next_idx
            while True:
                async with idx_lock:
                    if next_idx >= total:
                        return
                    idx = next_idx
                    next_idx += 1

                item = items[idx]
                async with sem:
                    result = await fn(*item) if isinstance(item, tuple) else await fn(item)

                batch_ids: list[int] | None = None
                batch_out: list[Any] | None = None
                local_completed = 0

                async with state_lock:
                    completed += 1
                    local_completed = completed
                    pending_ids.append(idx)
                    pending_out[idx] = result

                    flush_now = completed % self.checkpoint_interval == 0 or completed == total
                    if flush_now and on_batch_done:
                        batch_ids = pending_ids
                        pending_ids = []
                        batch_out = [pending_out.pop(i) for i in batch_ids]

                if batch_ids is not None and batch_out is not None and on_batch_done:
                    await on_batch_done(batch_out, batch_ids)
                    if total_progress_bar is not None:
                        total_progress_bar.update(len(batch_ids))
                    if pbar is not None:
                        remaining = total - local_completed
                        pbar.reset(total=min(remaining, self.checkpoint_interval))

                if pbar is not None:
                    pbar.update(1)
                if on_progress:
                    await on_progress(local_completed, total)
                if on_batch_progress:
                    batch_total = min(self.checkpoint_interval, total)
                    await on_batch_progress(pbar.n if pbar is not None else 0, batch_total)

        await asyncio.gather(*(worker() for _ in range(self.max_concurrency)))
        self._finalize_progress(total_progress_bar, pbar)

    @staticmethod
    def _init_queue(items):
        """Initialize an async queue with (index, item) pairs."""
        queue: asyncio.Queue[tuple[int, object]] = asyncio.Queue()
        for idx, item in enumerate(items):
            queue.put_nowait((idx, item))
        return queue

    def _wrap_retry(self, fn):
        """Wrap a coroutine function with retry/backoff policy."""
        return retry(
            wait=wait_exponential(multiplier=1, min=4, max=10),
            stop=stop_after_attempt(self.retries),
            reraise=True,
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )(fn)

    def _init_progress(self, progress: bool, total: int):
        """Initialize total and batch progress bars."""
        if not progress:
            return None, None
        thread_name = threading.current_thread().name
        total_progress_bar = tqdm(
            total=total,
            desc=f"[{thread_name}] Processing all items in worker",
            leave=True,
            unit="sample",
        )
        pbar = tqdm(
            total=min(self.checkpoint_interval, total),
            leave=True,
            desc=f"[{thread_name}] Processing batch",
            unit="sample",
        )
        return total_progress_bar, pbar

    async def _after_item(
        self,
        *,
        completed: int,
        pending_ids: list[int],
        idx: int,
        out,
        on_batch_done,
        on_progress,
        on_batch_progress,
        total: int,
        queue_size: int,
        total_progress_bar,
        pbar,
    ):
        """Update counters, trigger callbacks, and update progress after one item."""
        completed += 1
        pending_ids.append(idx)

        flush_now = completed % self.checkpoint_interval == 0 or completed == total
        if flush_now and on_batch_done:
            await on_batch_done(out, pending_ids)
            if total_progress_bar is not None:
                total_progress_bar.update(len(pending_ids))
            pending_ids = []
            if pbar is not None:
                pbar.reset(total=min(queue_size, self.checkpoint_interval))

        if pbar is not None:
            pbar.update(1)
        if on_progress:
            await on_progress(completed, total)
        if on_batch_progress:
            batch_total = min(self.checkpoint_interval, total)
            await on_batch_progress(pbar.n if pbar is not None else 0, batch_total)

        return completed, pending_ids

    @staticmethod
    def _finalize_progress(total_progress_bar, pbar):
        """Close progress bars if they were initialized."""
        if pbar is not None:
            pbar.close()
        if total_progress_bar is not None:
            total_progress_bar.close()
