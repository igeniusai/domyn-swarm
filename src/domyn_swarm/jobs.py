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

import logging
import os
import asyncio
import dataclasses
import abc
import threading
from typing import Callable, Coroutine, Dict, List, Sequence, Any, Tuple
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)
from tqdm.asyncio import tqdm

import pandas as pd
from rich.console import Console
from openai.types.chat.chat_completion import Choice, ChatCompletion
from openai import NOT_GIVEN

from domyn_swarm.helpers import (
    compute_perplexity_metrics,
    extract_token_logprobs,
    setup_logger,
)

logger = setup_logger(__name__, level=logging.INFO)


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
        timeout: float = NOT_GIVEN,
        **extra_kwargs,
    ):
        from openai import AsyncOpenAI

        self.endpoint = endpoint or os.getenv("ENDPOINT")
        if not self.endpoint:
            raise RuntimeError("ENDPOINT env-var not set")
        self.model: str = model
        self.batch_size: int = batch_size
        self.parallel: int = parallel
        self.retries: int = retries
        self.timeout: float = timeout
        self.input_column_name: str = input_column_name
        self.output_column_name: str = output_column_name
        self.client = AsyncOpenAI(
            base_url=f"{self.endpoint}/v1", api_key="-", organization="-", project="-", timeout=timeout
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

        # ───── restore checkpoint ───────────────────────────────────────────────
        if os.path.exists(ckp_path):
            done_df = pd.read_parquet(ckp_path)
            done_idx = set(done_df.index)
            logger.info(f"[ckp] resuming – {len(done_df)} rows done already")
        else:
            done_df = pd.DataFrame()
            done_idx = set()

        todo_df = df.loc[~df.index.isin(done_idx)].copy()
        idx_map = todo_df.index.to_numpy()  # position → original index

        pbar = tqdm(
            total=len(df),
            initial=len(done_df),
            desc=f"[{threading.current_thread().name}] Total samples processed",
            dynamic_ncols=True,
        )

        # ───── checkpoint flush callback ────────────────────────────────────────
        async def _flush(out_list: list, new_ids: list[int]) -> None:
            nonlocal done_df
            global_indices = [idx_map[i] for i in new_ids]  # restore originals
            tmp = df.loc[global_indices].copy()

            if isinstance(self.output_column_name, str):
                tmp[self.output_column_name] = [out_list[i] for i in new_ids]
            else:
                for col_idx, col_name in enumerate(self.output_column_name):
                    tmp[col_name] = [out_list[i][col_idx] for i in new_ids]

            done_df = pd.concat([done_df, tmp])  # keep original idx
            await asyncio.to_thread(done_df.to_parquet, ckp_path)

            pbar.update(len(new_ids))
            logger.info(f"[ckp] wrote {len(done_df)}/{len(df)} rows")

        self._ckp_flush = _flush  # type: ignore[attr-defined]

        try:
            await self.transform(todo_df)  # processing happens
        finally:
            if hasattr(self, "_ckp_flush"):
                delattr(self, "_ckp_flush")

        # no combine_first ⇒ no duplicates
        try:
            os.remove(ckp_path)
        except FileNotFoundError:
            pass

        return done_df.sort_index()  # 50 rows, in order

    async def batched(
        self,
        seq: Sequence,
        fn: Coroutine,
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
        # allow explicit override or fall back to checkpoint flush
        on_batch_done = on_batch_done or getattr(self, "_ckp_flush", None)

        fn = retry(
            wait=wait_exponential(multiplier=1, min=4, max=10),
            stop=stop_after_attempt(self.retries),
            reraise=True,
            before_sleep=before_sleep_log(logger=logger, log_level=logging.WARN),
        )(fn)

        out: list[Any | None] = [None] * len(seq)
        sem = asyncio.Semaphore(self.parallel)
        queue: asyncio.Queue[tuple[int, Any]] = asyncio.Queue()

        for idx, item in enumerate(seq):
            queue.put_nowait((idx, item))

        lock = asyncio.Lock()
        completed = 0
        pending_ids: list[int] = []

        pbar = tqdm(
            total=min(self.batch_size, len(seq)),
            desc=f"[{threading.current_thread().name}] Batch request execution",
            dynamic_ncols=True,
            leave=True,
        )

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
                    pbar.update(1)
                    pending_ids.append(idx)

                    flush_now = completed % self.batch_size == 0 or completed == len(
                        seq
                    )
                    if flush_now and on_batch_done:
                        ids_now = pending_ids
                        pending_ids = []
                        await on_batch_done(out, ids_now)
                        remaining = (
                            self.batch_size
                            if queue.qsize() >= self.batch_size
                            else queue.qsize()
                        )
                        pbar.reset(total=remaining)

        try:
            await tqdm.gather(
                *(asyncio.create_task(worker()) for _ in range(self.parallel)),
                desc=f"[{threading.current_thread().name}] Worker task completion",
            )
        except Exception as e:
            logger.info(f"An exception occurred while the worker was running: {e}")
        finally:
            pbar.close()

        return out

    @abc.abstractmethod
    async def transform(self, df: pd.DataFrame):
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
        timeout=NOT_GIVEN,
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
            timeout=timeout,
            **extra_kwargs,
        )

    async def transform(self, df: pd.DataFrame):
        df = df.copy()

        async def _call(prompt: str) -> str:
            from openai.types.completion import Completion

            resp: Completion = await self.client.completions.create(
                model=self.model, prompt=prompt, extra_body=self.kwargs
            )
            return resp.choices[0].text

        await self.batched(df[self.input_column_name].tolist(), _call)


class ChatCompletionJob(SwarmJob):
    """
    Input DF must have column `messages` (list of dicts).
    Output DF gets `answer`.
    """

    async def transform(self, df: pd.DataFrame):
        from openai.types.chat import ChatCompletion

        df = df.copy()

        async def _call(messages: list[dict]) -> str:
            resp: ChatCompletion = await self.client.chat.completions.create(
                model=self.model, messages=messages, extra_body=self.kwargs
            )
            return resp.choices[0].message.content

        await self.batched(
            [messages for messages in df[self.input_column_name].tolist()], _call
        )


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

    async def transform(self, df: pd.DataFrame):
        from openai.types.chat import ChatCompletion

        df = df.copy()

        async def _call(messages) -> list[str]:
            """Return *n* completions for one prompt."""
            resp: ChatCompletion = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                n=self.n,  # ask the API for n choices at once
                extra_body=self.kwargs,
            )
            return [choice.message.content for choice in resp.choices]

        # _batched now returns List[List[str]] (len == n for each inner list)
        await self.batched(
            [messages for messages in df[self.input_column_name].tolist()],
            _call,
        )


class PerplexityMixin:
    def compute_from_choice(self, choice: Choice) -> Tuple[float, float]:
        token_logprobs = extract_token_logprobs(choice)
        perp, bottom50 = compute_perplexity_metrics(token_logprobs)
        return perp, bottom50


class ChatCompletionPerplexityJob(PerplexityMixin, SwarmJob):
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

    async def transform(self, df: pd.DataFrame):
        df = df.copy()

        async def _call(messages) -> dict:
            resp: ChatCompletion = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                logprobs=True,
                extra_body=self.kwargs,
            )

            choice: Choice = resp.choices[0]
            text = choice.message.content

            return text, *self.compute_from_choice(choice)

        _ = await self.batched(
            [messages for messages in df[self.input_column_name].tolist()], _call
        )


class MultiTurnChatCompletionJob(SwarmJob):
    """
    For each row’s `messages` (a list of dicts), replay the conversation
    turn by turn, appending the assistant’s reply after each user/tool message.

    - Input  column: `messages`
    - Output column: `running_messages`
    """

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        model: str = "",
        input_column_name: str = "messages",
        output_column_name: str = "results",
        batch_size: int = 16,
        parallel: int = 2,
        retries: int = 5,
        **extra_kwargs: Any,
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

    async def transform(self, df: pd.DataFrame):
        # Copy input
        df = df.copy()

        await self.batched(
            df[self.input_column_name].tolist(),
            self._run_multi_turn,
        )

    async def _run_multi_turn(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        # Find indices of user or tool messages
        user_idxs = [i for i, m in enumerate(messages) if m["role"] in {"user", "tool"}]
        if not user_idxs:
            raise ValueError("Input must contain at least one 'user' or 'tool' message")

        running: List[Dict[str, Any]] = []

        # Zip together slice starts and ends
        idx = 0

        for i in user_idxs:
            # append all the messages until the next user message (including system, etc.)
            running.extend(messages[idx : i + 1])

            resp: ChatCompletion = await self.client.chat.completions.create(
                model=self.model, messages=running, extra_body=self.kwargs
            )
            choice = resp.choices[0]

            # append the assistant's response to the messages
            response_dict = {
                "role": "assistant",
                "content": choice.message.content,
            }
            if hasattr(choice.message, "reasoning_content"):
                response_dict["reasoning_content"] = choice.message.reasoning_content
            running.append(response_dict)

            # update the index skipping assistant message
            idx = i + 2
        return running
