# swarm_job.py  ───────────────────────────────────────────────────────────────
"""
Light-weight framework for driver scripts that run **inside** a Domyn swarm.
Every class:

1.  Reads the load-balancer URL from the `ENDPOINT` env-var (injected by
    `DomynLLMSwarm` on the head node).
2.  Creates a single `openai.AsyncOpenAI` client pointing to that URL
    (`base_url=ENDPOINT`, `api_key="-"`).
3.  Provides `.run(df)` – a *synchronous* wrapper around an async
    coroutine so users don’t have to think about `asyncio` unless they
    want to.
4.  Implements `.to_kwargs()` ⇒ JSON-serialisable dict so the object can
    be reconstructed by `domyn_swarm.run_job` inside the allocation.

Sub-classes included:

* `CompletionJob`       → one prompt → one text completion
* `ChatCompletionJob`   → list-of-messages → one assistant reply
* `EmbeddingJob`        → one text → embedding vector
"""

import os, asyncio, inspect, json, dataclasses, abc
from typing import Iterable, Awaitable, Sequence, Any

import pandas as pd
from openai import AsyncOpenAI


# ────────────────────────────────────────────────────────────────────────────
#  Base class
# ────────────────────────────────────────────────────────────────────────────
class SwarmJob(abc.ABC):
    """
    Sub-class this and implement `async transform(self, df) -> DataFrame`.

    Use `.run(df)` from your driver to execute synchronously *or* call
    `await job.transform(df)` directly in an async driver.
    """

    # constructor arguments all get forwarded by `submit_job`
    def __init__(
        self,
        *,
        endpoint: str | None = None,
        model: str = "",
        batch_size: int = 32,
        **kwargs,
    ):
        self.endpoint = endpoint or os.getenv("ENDPOINT")
        self.model = model or os.getenv("MODEL", "")
        if not self.endpoint:
            raise RuntimeError("ENDPOINT env-var is not set")

        self.batch_size = batch_size
        self.client     = AsyncOpenAI(base_url=self.endpoint, api_key="-", organization="-", project="-")
        self.kwargs     = kwargs      # free-form tuner params (temperature …)

    # ------------- public sync façade --------------------------------------
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        coro = self.transform(df)
        return asyncio.run(coro) if inspect.iscoroutine(coro) else coro

    # ------------- serialisation helper ------------------------------------
    def to_kwargs(self) -> dict:
        if dataclasses.is_dataclass(self):
            base = dataclasses.asdict(self)
        else:
            base = {
                k: v
                for k, v in self.__dict__.items()
                if isinstance(v, (str, int, float, bool, list, dict, type(None)))
            }
        # Never serialise the live client object!
        base.pop("client", None)
        return base

    # ------------- batching helper -----------------------------------------
    async def _batched(
        self,
        seq: Sequence[Any],
        coro_fn,                         # takes one item, returns awaitable
    ) -> list[Any]:
        out: list[Any] = []
        for i in range(0, len(seq), self.batch_size):
            chunk = seq[i : i + self.batch_size]
            out += await asyncio.gather(*(coro_fn(x) for x in chunk))
        return out
    
    async def _checkpoint(self, df: pd.DataFrame, path: str) -> None:
        """
        Save the DataFrame to a Parquet file at `path`.
        This is useful for long-running jobs that may need to be resumed.
        """
        df.to_parquet(path, index=False)
        print(f"Checkpoint saved to {path}")

    # ------------- user must implement -------------------------------------
    @abc.abstractmethod
    async def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ...


# ────────────────────────────────────────────────────────────────────────────
#  Sub-classes
# ────────────────────────────────────────────────────────────────────────────
@dataclasses.dataclass
class CompletionJob(SwarmJob):
    """
    Expects a column `prompt`, writes a column `completion`.
    Extra kwargs (temperature, max_tokens, …) are passed through.
    """

    async def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        async def _one(prompt: str) -> str:
            resp = await self.client.completions.create(
                model=self.model,
                prompt=prompt,
                **self.kwargs,
            )
            return resp.choices[0].text

        df["completion"] = await self._batched(df["prompt"].tolist(), _one)
        return df


@dataclasses.dataclass
class ChatCompletionJob(SwarmJob):
    """
    Column `messages` must hold a list[dict] in the OpenAI chat format.
    Writes column `answer`.
    """

    async def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        async def _one(msgs) -> str:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=msgs,
                **self.kwargs,
            )
            return resp.choices[0].message.content

        df["answer"] = await self._batched(df["messages"].tolist(), _one)
        return df


@dataclasses.dataclass
class EmbeddingJob(SwarmJob):
    """
    Column `text`, writes column `embedding` (list[float]).
    """

    async def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        async def _one(txt: str):
            resp = await self.client.embeddings.create(
                input=txt,
                model=self.model,           # model name ignored by vLLM embedding route
            )
            return resp.data[0].embedding

        df["embedding"] = await self._batched(df["text"].tolist(), _one)
        return df