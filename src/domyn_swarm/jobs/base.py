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
import os
from typing import Callable

import pandas as pd
from openai import AsyncOpenAI
from tqdm import tqdm

from .batching import BatchExecutor
from .checkpointing import CheckpointManager


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
        batch_size: int = 16,
        parallel: int = 2,
        retries: int = 5,
        timeout: float = 600,
        client=None,
        client_kwargs: dict = None,
        **extra_kwargs,
    ):
        self.endpoint = endpoint or os.getenv("ENDPOINT")
        if not self.endpoint:
            raise RuntimeError("ENDPOINT environment variable is not set")

        self.model = model
        self.provider = provider
        self.input_column_name = input_column_name
        self.output_column_name = output_column_name
        self.batch_size = batch_size
        self.parallel = parallel
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
            manager.flush(out_list, new_ids, self.output_column_name, idx_map)
            tqdm.write(f"[ckp] flushed {len(new_ids)} rows")

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
        executor = BatchExecutor(self.parallel, self.batch_size, self.retries)
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
