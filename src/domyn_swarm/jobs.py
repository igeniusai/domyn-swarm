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
* `EmbeddingJob`        → one text → embedding vector
"""
import os
import asyncio
import hashlib
import dataclasses
import abc
from typing import Sequence, Any

import pandas as pd
from openai import AsyncOpenAI, OpenAIError
from openai.types.chat import ChatCompletion
from openai.types.completion import Completion
from openai.types.create_embedding_response import CreateEmbeddingResponse
from rich import print as rprint

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
        batch_size: int = 32,
        parallel: int = 32,
        retries: int = 5,
        **extra_kwargs,
    ):
        self.endpoint   = endpoint or os.getenv("ENDPOINT")
        if not self.endpoint:
            raise RuntimeError("ENDPOINT env-var not set")
        self.model = model
        self.batch_size = batch_size
        self.parallel   = parallel
        self.retries    = retries
        self.client     = AsyncOpenAI(base_url=f"{self.endpoint}/v1", api_key="-", organization="-", project="-")
        self.kwargs     = {**extra_kwargs.get("kwargs", {})}

    def run(self, df: pd.DataFrame, checkpoint_dir: str = ".checkpoints") -> pd.DataFrame:
        """Synchronous entry-point: runs the async pipeline end-to-end."""
        return asyncio.run(self._run_async(df, checkpoint_dir))

    async def _run_async(self, df: pd.DataFrame, ckp_dir: str) -> pd.DataFrame:
        """Slice-based checkpointing + transform calls."""
        os.makedirs(ckp_dir, exist_ok=True)
        # create stable tag from DataFrame contents
        tag   = hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()[:8]
        ckp_p = os.path.join(ckp_dir, f"{self.__class__.__name__}_{tag}.parquet")

        if os.path.exists(ckp_p):
            done_df       = pd.read_parquet(ckp_p)
            processed_idx = set(done_df.index)
            rprint(f"[ckp] resuming, {len(done_df)} rows done")
        else:
            done_df       = pd.DataFrame()
            processed_idx = set()

        todo_df = df.loc[~df.index.isin(processed_idx)]

        while not todo_df.empty:
            slice_df = todo_df.iloc[:self.batch_size]
            result   = await self.transform(slice_df)
            done_df  = pd.concat([done_df, result]).sort_index()
            done_df.to_parquet(ckp_p)
            rprint(f"[ckp] wrote {len(done_df)}/{len(df)} rows")
            todo_df  = todo_df.iloc[self.batch_size:]
        
        os.remove(ckp_p)  # remove checkpoint file after processing

        return done_df

    async def _batched(
        self,
        seq: Sequence[Any],
        fn,  # callable(item) -> Awaitable[result]
    ) -> list[Any]:
        """
        Call fn(item) for each item in `seq`, up to `parallel` tasks at once,
        retrying each up to `retries` times with exponential backoff.
        """
        out = [None] * len(seq)

        async def _worker(idx: int, item: Any):
            for attempt in range(self.retries + 1):
                try:
                    out[idx] = await fn(item)
                    return
                except (OpenAIError, asyncio.TimeoutError) as e:
                    if attempt == self.retries:
                        raise
                    backoff = 2 ** (attempt + 1)
                    rprint(f"[retry] {e!r}, attempt {attempt+1}/{self.retries}, sleep {backoff}s")
                    await asyncio.sleep(backoff)

        for start in range(0, len(seq), self.parallel):
            batch = seq[start : start + self.parallel]
            tasks = [
                asyncio.create_task(_worker(start + i, batch[i]))
                for i in range(len(batch))
            ]
            await asyncio.gather(*tasks)

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
            if isinstance(v, (str, int, float, bool, list, dict, type(None))) and k not in {"endpoint", "model"}
        }


class CompletionJob(SwarmJob):
    """
    Input DF must have column `prompt`; output DF gets `completion`.
    """

    async def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        async def _call(prompt: str) -> str:
            resp: Completion = await self.client.completions.create(
                model=self.model, prompt=prompt, **self.kwargs
            )
            return resp.choices[0].text

        df["completion"] = await self._batched(df["prompt"].tolist(), _call)
        return df


class ChatCompletionJob(SwarmJob):
    """
    Input DF must have column `messages` (list of dicts).
    Output DF gets `answer`.
    """

    async def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        async def _call(messages) -> str:
            resp: ChatCompletion = await self.client.chat.completions.create(
                model=self.model, messages=messages, **self.kwargs
            )
            return resp.choices[0].message.content

        df["answer"] = await self._batched([[message] for message in df["messages"].tolist()], _call)
        return df

