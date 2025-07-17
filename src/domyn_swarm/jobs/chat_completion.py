from typing import Any, Dict, List, Tuple

import pandas as pd
from openai.types.chat.chat_completion import ChatCompletion, Choice

from domyn_swarm.helpers.data import (
    compute_perplexity_metrics,
    extract_token_logprobs,
)
from domyn_swarm.jobs.base import SwarmJob


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
        timeout=600,
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
        self.output_column_name = ["text", "perplexity", "bottom50_perplexity"]

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
