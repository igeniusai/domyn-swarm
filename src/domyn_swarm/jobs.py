# swarm_job.py  ───────────────────────────────────────────────────────────────
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
    be reconstructed by `domyn_swarm.run_job` inside the allocation.

Sub-classes included:

* `CompletionJob`       → one prompt → one text completion
* `ChatCompletionJob`   → list-of-messages → one assistant reply
"""

import os
import asyncio
import dataclasses
import abc
from typing import Callable, List, Sequence, Any
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import pandas as pd
from rich import print as rprint

from domyn_swarm.helpers import compute_perplexity


class SwarmJob(abc.ABC):
    """
    Base class for jobs that run inside a Domyn swarm allocation.
    - Reads ENDPOINT env-var for the load-balancer URL.
    - Creates one AsyncOpenAI client.
    - Offers .run(df) as a sync façade over an async pipeline.
    - Handles slice-based checkpointing.
    - Batches up to `parallel` concurrent calls with retries.
    """

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        model: str = "",
        input_column_name: str = "messages",
        output_column_name: str | list = "result",
        batch_size: int = 16,
        parallel: int = 2,
        retries: int = 5,
        **extra_kwargs,
    ):
        from openai import AsyncOpenAI

        self.endpoint = endpoint or os.getenv("ENDPOINT")
        if not self.endpoint:
            raise RuntimeError("ENDPOINT env-var not set")
        self.model = model
        self.batch_size = batch_size
        self.parallel = parallel
        self.retries = retries
        self.input_column_name = input_column_name
        self.output_column_name = output_column_name
        self.client = AsyncOpenAI(
            base_url=f"{self.endpoint}/v1", api_key="-", organization="-", project="-"
        )
        if "kwargs" in extra_kwargs.keys():
            self.kwargs = {**extra_kwargs.get("kwargs", {})}
        else:
            self.kwargs = {**extra_kwargs}

    async def run(
        self, df: pd.DataFrame, tag: str, checkpoint_dir: str = ".checkpoints"
    ) -> pd.DataFrame:
        """Synchronous entry-point: runs the async pipeline end-to-end."""
        return await self._run_async(df, tag, checkpoint_dir)

    async def _run_async(
        self,
        df: pd.DataFrame,
        tag: str,
        ckp_dir: str,
    ) -> pd.DataFrame:
        """
        High-level orchestration.

        Loads an existing checkpoint (if any), builds a hidden flush callback
        `self._ckp_flush`, runs `transform()` once, and finally returns the
        combined DataFrame.
        """
        os.makedirs(ckp_dir, exist_ok=True)
        ckp_path = os.path.join(ckp_dir, f"{self.__class__.__name__}_{tag}.parquet")

        # ───────────── restore previous progress (if any)
        if os.path.exists(ckp_path):
            done_df = pd.read_parquet(ckp_path)
            done_idx = set(done_df.index)
            rprint(f"[ckp] resuming – {len(done_df)} rows done already")
        else:
            done_df = pd.DataFrame()
            done_idx = set()

        todo_df = df.loc[~df.index.isin(done_idx)]
        idx_map: List[Any] = todo_df.index.tolist()  # position → global index

        # ───────────── checkpoint flush callback (captured by _batched)
        async def _flush(out_list: list, new_ids: list[int]) -> None:
            nonlocal done_df
            global_rows = [idx_map[i] for i in new_ids]
            tmp = todo_df.loc[global_rows].copy()

            if isinstance(self.output_column_name, str):
                tmp[self.output_column_name] = [out_list[i] for i in new_ids]
            else:
                for col_idx, col_name in enumerate(self.output_column_name):
                    tmp[col_name] = [out_list[i][col_idx] for i in new_ids]

            done_df = pd.concat([done_df, tmp]).sort_index()
            done_df.to_parquet(ckp_path)
            rprint(f"[ckp] wrote {len(done_df)}/{len(df)} rows")

        # expose the callback so _batched() can discover it transparently
        self._ckp_flush = _flush  # type: ignore[attr-defined]

        try:
            _ = await self.transform(todo_df)  # returns df, but we rely on flushes
        finally:
            # always remove the attribute, even if transform() raises
            if hasattr(self, "_ckp_flush"):
                delattr(self, "_ckp_flush")

        # All slices flushed – combine w/ any remainder produced by transform()
        final_df = done_df.combine_first(todo_df).sort_index()

        # Success → drop checkpoint file
        try:
            os.remove(ckp_path)
        except FileNotFoundError:
            pass

        return final_df

    async def batched(
        self,
        seq: Sequence,
        fn: Callable,
        *,
        on_batch_done: Callable[[list, list[int]], None] | None = None,
    ) -> list:
        """
        Run *fn* concurrently over *seq* with retries.

        • Respects `self.parallel` for concurrency.
        • Retries `openai.APITimeoutError` up to `self.retries` times.
        • Calls *on_batch_done* (or the hidden self._ckp_flush) every
          `self.batch_size` completions and once at the end.
        """
        from openai import APITimeoutError

        # allow explicit override or fall back to checkpoint flush
        on_batch_done = on_batch_done or getattr(self, "_ckp_flush", None)

        fn = retry(
            fn,
            retry=retry_if_exception_type(APITimeoutError),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            stop=stop_after_attempt(self.retries),
        )

        out: list[Any | None] = [None] * len(seq)
        sem = asyncio.Semaphore(self.parallel)
        queue: asyncio.Queue[tuple[int, Any]] = asyncio.Queue()

        for idx, item in enumerate(seq):
            queue.put_nowait((idx, item))

        lock = asyncio.Lock()
        completed = 0
        pending_ids: list[int] = []

        async def worker() -> None:
            nonlocal completed, pending_ids
            while not queue.empty():
                idx, item = await queue.get()
                async with sem:
                    if isinstance(item, tuple):
                        out[idx] = await fn(*item)
                    else:
                        out[idx] = await fn(item)

                async with lock:
                    completed += 1
                    pending_ids.append(idx)

                    flush_now = completed % self.batch_size == 0 or completed == len(
                        seq
                    )
                    if flush_now and on_batch_done:
                        ids_now = pending_ids
                        pending_ids = []
                        await on_batch_done(out, ids_now)

        await asyncio.gather(
            *(asyncio.create_task(worker()) for _ in range(self.parallel))
        )
        return out

    @abc.abstractmethod
    async def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process a slice of the DataFrame and return same-shaped DataFrame."""
        ...

    def to_kwargs(self) -> dict:
        """Serialize constructor args for remote reconstruction."""
        if dataclasses.is_dataclass(self):
            return dataclasses.asdict(self)
        return {
            k: v
            for k, v in self.__dict__.items()
            if isinstance(v, (str, int, float, bool, list, dict, type(None)))
            and k not in {"endpoint", "model"}
        }


class CompletionJob(SwarmJob):
    """
    Input DF must have column `prompt`; output DF gets `completion`.
    """

    def __init__(
        self,
        *,
        endpoint=None,
        model="",
        input_column_name="prompt",
        output_column_name="completion",
        batch_size=16,
        parallel=2,
        retries=5,
        **extra_kwargs,
    ):
        super().__init__(
            endpoint=endpoint,
            model=model,
            input_column_name=input_column_name,
            output_column_name=output_column_name,
            batch_size=batch_size,
            parallel=parallel,
            retries=retries,
            **extra_kwargs,
        )

    async def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        async def _call(prompt: str) -> str:
            from openai.types.completion import Completion

            resp: Completion = await self.client.completions.create(
                model=self.model, prompt=prompt, **self.kwargs
            )
            return resp.choices[0].text

        df[self.output_column_name] = await self.batched(
            df[self.input_column_name].tolist(), _call
        )
        return df


class ChatCompletionJob(SwarmJob):
    """
    Input DF must have column `messages` (list of dicts).
    Output DF gets `answer`.
    """

    async def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        from openai.types.chat import ChatCompletion

        df = df.copy()

        async def _call(messages) -> str:
            resp: ChatCompletion = await self.client.chat.completions.create(
                model=self.model, messages=messages, **self.kwargs
            )
            return resp.choices[0].message.content

        df[self.output_column_name] = await self.batched(
            [[message] for message in df[self.input_column_name].tolist()], _call
        )
        return df


class MultiChatCompletionJob(SwarmJob):
    """
    Produce *n* independent chat completions for every row.

    Input  column : `messages`
    Output columns: `generated_1`, `generated_2`, …, `generated_n`
    """

    def __init__(self, n: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.n = n
        base_output_column_name = kwargs.pop("output_column_name")
        self.output_column_name = (
            [f"{base_output_column_name}_{i + 1}" for i in range(n)]
            if isinstance(base_output_column_name, str)
            else base_output_column_name
        )

    async def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        from openai.types.chat import ChatCompletion

        df = df.copy()

        async def _call(messages) -> list[str]:
            """Return *n* completions for one prompt."""
            resp: ChatCompletion = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                n=self.n,  # ask the API for n choices at once
                **self.kwargs,
            )
            return [choice.message.content for choice in resp.choices]

        # _batched now returns List[List[str]] (len == n for each inner list)
        multi_outputs = await self.batched(
            [[m] for m in df[self.input_column_name].tolist()],
            _call,
        )

        # Unpack the list-of-lists into separate DataFrame columns
        for i, col in enumerate(self.output_column_name):
            df[col] = [row[i] for row in multi_outputs]

        return df


class ChatCompletionPerplexityJob(SwarmJob):
    def __init__(
        self,
        *,
        endpoint=None,
        model="",
        input_column_name="messages",
        output_column_name="result",
        batch_size=16,
        parallel=2,
        retries=5,
        **extra_kwargs,
    ):
        super().__init__(
            endpoint=endpoint,
            model=model,
            input_column_name=input_column_name,
            output_column_name=output_column_name,
            batch_size=batch_size,
            parallel=parallel,
            retries=retries,
            **extra_kwargs,
        )
        self.output_column_name = ["text", "perplexity", " bottom50_perplexity"]

    async def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        from openai.types.chat.chat_completion import ChatCompletion, Choice

        df = df.copy()

        async def _call(messages) -> dict:
            resp: ChatCompletion = await self.client.chat.completions.create(
                model=self.model, messages=messages, logprobs=True, **self.kwargs
            )

            choice: Choice = resp.choices[0]
            text = choice.message.content

            # Handle logprobs from modern schema
            token_logprobs = []
            if choice.logprobs and choice.logprobs.content:
                token_logprobs: list[float] = [
                    token_logprob.logprob
                    for token_logprob in choice.logprobs.content
                    if token_logprob.logprob is not None
                ]

            bottom_50 = sorted(token_logprobs)[:50]
            perplexity = compute_perplexity(token_logprobs)
            bottom_50_perplexity = compute_perplexity(bottom_50)

            return text, perplexity, bottom_50_perplexity

        _ = await self.batched(
            [[message] for message in df[["prompt", "rewa"]].tolist()], _call
        )

        return df
