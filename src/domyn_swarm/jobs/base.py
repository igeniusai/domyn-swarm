"""
Light-weight framework for driver scripts that run **inside** a Domyn swarm.
Every class:

1.  Reads the load-balancer URL from the `ENDPOINT` env-var (injected by
    `DomynLLMSwarm` on the head node).
2.  Creates a single `openai.AsyncOpenAI` client pointing to that URL
    (`base_url=ENDPOINT`, `api_key="-"`).
3.  Provides `.run(df)` - a *synchronous* wrapper around an async
    coroutine so users don't have to think about `asyncio` unless they
    want to.
4.  Implements `.to_kwargs()` ⇒ JSON-serialisable dict so the object can
    be reconstructed by `domyn_swarm.jobs.run` inside the allocation.

Sub-classes included:

* `CompletionJob`       → one prompt → one text completion
* `ChatCompletionJob`   → list-of-messages → one assistant reply
"""

import abc
import dataclasses
import logging
import os
import threading
from typing import Callable

import pandas as pd
from openai import AsyncOpenAI
from tqdm import tqdm

from ..helpers.logger import setup_logger
from .batching import BatchExecutor
from .checkpointing import CheckpointManager

logger = setup_logger("domyn_swarm.jobs.base", level=logging.INFO)


class SwarmJob(abc.ABC):
    """
    Abstract base class for jobs running in a Domyn LLM swarm.

    Key features:
    - Handles checkpointing with automatic recovery.
    - Provides batching with retry and callback hooks.
    - Uses a pluggable LLM client (OpenAI, etc.) via a factory.
    """

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        model: str = "",
        provider: str = "openai",
        input_column_name: str = "messages",
        output_column_name: str | list = "result",
        checkpoint_interval: int = 16,
        # TODO: deprecated, remove in future versions
        batch_size: int | None = None,
        # TODO: deprecated, remove in future versions
        parallel: int | None = None,
        max_concurrency: int = 2,
        retries: int = 5,
        timeout: float = 600,
        client=None,
        client_kwargs: dict | None = None,
        **extra_kwargs,
    ):
        """
        Initialize the job with parameters and an optional LLM client.

        Parameters:
            endpoint: Optional LLM endpoint URL (overrides `ENDPOINT` env var).
            model: Model name to use (e.g., "gpt-4").
            provider: LLM provider (default: "openai").
            input_column_name: Name of the input column in the DataFrame.
            output_column_name: Name of the output column(s) in the DataFrame.
            batch_size: Size of each batch for processing. (deprecated, use `checkpoint_interval`).
            checkpoint_interval: Number of items to process before checkpointing.
            parallel: Number of concurrent requests to process. (deprecated, use `max_concurrency`).
            max_concurrency: Maximum number of concurrent requests to process.
            retries: Number of retries for failed requests.
            timeout: Request timeout in seconds.
            client: Optional pre-initialized LLM client (e.g., `AsyncOpenAI`).
            client_kwargs: Additional kwargs for the LLM client.
            extra_kwargs: Additional parameters to pass to the job constructor.
        """
        self.endpoint = endpoint or os.getenv("ENDPOINT")
        if not self.endpoint:
            raise RuntimeError("ENDPOINT environment variable is not set")

        if not model:
            raise ValueError("Model name must be specified")

        if parallel is not None:
            logger.warning(
                "The `parallel` parameter is deprecated. Use `max_concurrent_requests` instead."
            )
            self.max_concurrency = parallel

        if batch_size is not None:
            logger.warning(
                "The `batch_size` parameter is deprecated. Use `checkpoint_interval` instead."
            )
            self.checkpoint_interval = batch_size

        self.model = model
        self.provider = provider
        self.input_column_name = input_column_name
        self.output_column_name = output_column_name
        self.checkpoint_interval = checkpoint_interval
        # TODO: deprecated, remove in future versions
        self.batch_size = batch_size
        # TODO: deprecated, remove in future versions
        self.parallel = parallel
        self.max_concurrency = max_concurrency
        self.retries = retries
        self.timeout = timeout
        self.kwargs = {**extra_kwargs.get("kwargs", extra_kwargs)}

        self.client = client or AsyncOpenAI(
            base_url=f"{self.endpoint}/v1",
            api_key="-",
            organization="-",
            project="-",
            timeout=timeout,
            **(client_kwargs or {}),
        )
        self._callbacks: dict[str, Callable] = {}

        self.results = None

    def register_callback(self, event: str, fn: Callable) -> None:
        """Register a named callback (e.g., 'on_batch_done')."""
        self._callbacks[event] = fn

    def get_callback(self, event: str) -> Callable | None:
        return self._callbacks.get(event)

    async def run(
        self,
        df: pd.DataFrame,
        tag: str,
        checkpoint_dir: str = ".checkpoints",
    ) -> pd.DataFrame:
        """
        Run the job end-to-end with checkpointing support.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, f"{self.__class__.__name__}_{tag}.parquet")
        manager = CheckpointManager(path, df)

        todo_df = manager.filter_todo()
        idx_map = todo_df.index.to_numpy()

        async def flush(out_list, new_ids):
            thread_name = threading.current_thread().name
            manager.flush(out_list, new_ids, self.output_column_name, idx_map)
            tqdm.write(
                f"[{thread_name}] Checkpoint flushed {len(new_ids)} rows, new total: {len(manager.done_df)}"
            )

        self.register_callback("on_batch_done", flush)

        try:
            await self.transform(todo_df)
        finally:
            self._callbacks.clear()

        self.results = manager.finalize()
        return self.results

    async def batched(self, seq: list, fn: Callable) -> list:
        """
        Run a batched async pipeline over `seq` using `fn`.

        Supports retrying and invokes the 'on_batch_done' callback if registered.
        """
        executor = BatchExecutor(
            self.max_concurrency, self.checkpoint_interval, self.retries
        )
        return await executor.run(
            seq, fn, on_batch_done=self.get_callback("on_batch_done")
        )

    @abc.abstractmethod
    async def transform(self, df: pd.DataFrame):
        """
        Subclasses must implement this to process a DataFrame slice.
        It should invoke `self.batched(...)` internally.
        """
        ...

    def to_kwargs(self) -> dict:
        """
        Serialize the job's constructor parameters (for remote reconstruction).
        """
        if dataclasses.is_dataclass(self):
            return dataclasses.asdict(self)

        return {
            k: v
            for k, v in self.__dict__.items()
            if isinstance(v, (str, int, float, bool, list, dict, type(None)))
            and k not in {"endpoint", "model", "client", "_callbacks"}
        }
