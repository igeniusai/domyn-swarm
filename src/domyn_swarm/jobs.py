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
        self.client = AsyncOpenAI(
            base_url=f"{self.endpoint}/v1", api_key="-", organization="-", project="-"
        )
        self.kwargs = {**extra_kwargs.get("kwargs", {})}

    async def run(
        self, df: pd.DataFrame, tag:str, checkpoint_dir: str = ".checkpoints"
    ) -> pd.DataFrame:
        """Synchronous entry-point: runs the async pipeline end-to-end."""
        return await self._run_async(df, tag, checkpoint_dir)

    async def _run_async(self, df: pd.DataFrame, tag: str, ckp_dir: str) -> pd.DataFrame:
        """Slice-based checkpointing + transform calls."""
        os.makedirs(ckp_dir, exist_ok=True)
        # create stable tag from DataFrame contents
        ckp_p = os.path.join(ckp_dir, f"{self.__class__.__name__}_{tag}.parquet")

        if os.path.exists(ckp_p):
            done_df = pd.read_parquet(ckp_p)
            processed_idx = set(done_df.index)
            rprint(f"[ckp] resuming, {len(done_df)} rows done")
        else:
            done_df = pd.DataFrame()
            processed_idx = set()

        todo_df = df.loc[~df.index.isin(processed_idx)]

        while not todo_df.empty:
            slice_df = todo_df.iloc[: self.batch_size]
            result = await self.transform(slice_df)
            done_df = pd.concat([done_df, result]).sort_index()
            done_df.to_parquet(ckp_p)
            rprint(f"[ckp] wrote {len(done_df)}/{len(df)} rows")
            todo_df = todo_df.iloc[self.batch_size :]

        os.remove(ckp_p)  # remove checkpoint file after processing

        return done_df

    async def _batched(self, seq, fn):
        out   = [None] * len(seq)
        sem   = asyncio.Semaphore(self.parallel)          # keeps *at most* N tasks alive
        queue = asyncio.Queue()

        for idx, item in enumerate(seq):
            await queue.put((idx, item))

        async def worker():
            while not queue.empty():
                idx, item = await queue.get()
                async with sem:
                    out[idx] = await fn(item)

        workers = [asyncio.create_task(worker()) for _ in range(self.parallel)]
        await asyncio.gather(*workers)
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

    async def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        async def _call(prompt: str) -> str:
            from openai.types.completion import Completion

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
        from openai.types.chat import ChatCompletion

        df = df.copy()

        async def _call(messages) -> str:
            resp: ChatCompletion = await self.client.chat.completions.create(
                model=self.model, messages=messages, **self.kwargs
            )
            return resp.choices[0].message.content

        df["answer"] = await self._batched(
            [[message] for message in df["messages"].tolist()], _call
        )
        return df
