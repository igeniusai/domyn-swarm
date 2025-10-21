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

from typing import Any, Dict, List, Tuple

import pandas as pd
from deprecated import deprecated
from openai.types.chat import ChatCompletionMessageParam
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

    api_version = 2

    def __init__(
        self,
        *,
        endpoint=None,
        model="",
        input_column_name="prompt",
        output_cols="completion",
        checkpoint_interval=16,
        max_concurrency=2,
        retries=5,
        timeout=600,
        **extra_kwargs,
    ):
        super().__init__(
            endpoint=endpoint,
            model=model,
            input_column_name=input_column_name,
            output_cols=output_cols,
            checkpoint_interval=checkpoint_interval,
            max_concurrency=max_concurrency,
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

    async def transform_items(self, items: list[str]) -> list[Any]:
        """
        Transform a list of prompts into a list of completions.
        Each item in `items` is a single prompt string.

        Args:
            items: List of prompt strings.

        Returns a list of completion strings.
        """
        outs = []
        for prompt in items:
            resp = await self.client.completions.create(
                model=self.model, prompt=prompt, extra_body=self.kwargs
            )
            outs.append(resp.choices[0].text)
        return outs


class ChatCompletionJob(SwarmJob):
    """
    Input DF must have column `messages` (list of dicts).
    Output DF gets `answer`.
    """

    api_version = 2

    def __init__(self, *, parse_reasoning: bool | None = None, **kwargs):
        explicit = parse_reasoning
        super().__init__(**kwargs)
        from_extras = bool(self.kwargs.pop("parse_reasoning", False))
        self.parse_reasoning = explicit if explicit is not None else from_extras
        self.output_cols = (
            ["result", "reasoning_content"] if self.parse_reasoning else "result"
        )

    @deprecated(
        "Use transform_items instead for better performance with small batches."
    )
    async def transform(self, df: pd.DataFrame):
        from openai.types.chat import ChatCompletion

        df = df.copy()

        async def _call(
            messages: list[ChatCompletionMessageParam],
        ) -> str | None | Tuple[str | None, str | None]:
            resp: ChatCompletion = await self.client.chat.completions.create(
                model=self.model, messages=messages, extra_body=self.kwargs
            )

            choice: Choice = resp.choices[0]
            if self.parse_reasoning:
                reasoning_content = getattr(choice.message, "reasoning_content", None)
                return choice.message.content, reasoning_content
            return choice.message.content

        await self.batched(
            [messages for messages in df[self.input_column_name].tolist()], _call
        )

    async def transform_items(
        self, items: list[list[ChatCompletionMessageParam]]
    ) -> list[Any]:
        outs = []
        for msgs in items:
            resp = await self.client.chat.completions.create(
                model=self.model, messages=msgs, extra_body=self.kwargs
            )
            choice = resp.choices[0]
            if self.parse_reasoning:
                outs.append(
                    {
                        "result": choice.message.content,
                        "reasoning_content": getattr(
                            choice.message, "reasoning_content", None
                        ),
                    }
                )
            else:
                outs.append(choice.message.content)
        return outs


class MultiChatCompletionJob(SwarmJob):
    """
    Produce *n* independent chat completions for every row.

    Input  column : `messages`
    Output columns: `generated_1`, `generated_2`, …, `generated_n`
    """

    api_version = 2

    def __init__(self, n: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.n = n
        base_output_cols = kwargs.pop("output_cols")
        self.output_cols = (
            [f"{base_output_cols}_{i + 1}" for i in range(n)]
            if isinstance(base_output_cols, str)
            else base_output_cols
        )

    @deprecated(
        "Use transform_items instead for better performance with small batches."
    )
    async def transform(self, df: pd.DataFrame):
        from openai.types.chat import ChatCompletion

        df = df.copy()

        async def _call(messages: list[ChatCompletionMessageParam]) -> list[Any]:
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

    async def transform_items(
        self, items: list[list[ChatCompletionMessageParam]]
    ) -> list[Any]:
        outs = []
        for msgs in items:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=msgs,
                n=self.n,  # ask the API for n choices at once
                extra_body=self.kwargs,
            )
            outs.append([choice.message.content for choice in resp.choices])
        return outs


class PerplexityMixin:
    def compute_from_choice(self, choice: Choice) -> Tuple[float, float]:
        token_logprobs = extract_token_logprobs(choice)
        perp, bottom50 = compute_perplexity_metrics(token_logprobs)
        return perp, bottom50


class ChatCompletionPerplexityJob(PerplexityMixin, SwarmJob):
    """
    For each row's `messages` (a list of dicts), produce a single chat
    completion with logprobs, and compute perplexity metrics.

    - Input  column: `messages`
    - Output columns: `text`, `perplexity`, `bottom50_perplexity`
    """

    api_version = 2

    def __init__(
        self,
        *,
        endpoint=None,
        model="",
        input_column_name="messages",
        output_cols="result",
        checkpoint_interval=16,
        max_concurrency=2,
        retries=5,
        **extra_kwargs,
    ):
        super().__init__(
            endpoint=endpoint,
            model=model,
            input_column_name=input_column_name,
            output_cols=output_cols,
            checkpoint_interval=checkpoint_interval,
            max_concurrency=max_concurrency,
            retries=retries,
            **extra_kwargs,
        )
        self.output_cols = ["text", "perplexity", "bottom50_perplexity"]

    @deprecated(
        "Use transform_items instead for better performance with small batches."
    )
    async def transform(self, df: pd.DataFrame):
        df = df.copy()

        async def _call(
            messages: list[ChatCompletionMessageParam],
        ) -> tuple[str | None, float, float]:
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

    async def transform_items(
        self, items: list[list[ChatCompletionMessageParam]]
    ) -> list[Any]:
        outs = []
        for msgs in items:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=msgs,
                logprobs=True,
                extra_body=self.kwargs,
            )
            choice = resp.choices[0]
            text = choice.message.content
            outs.append((*self.compute_from_choice(choice), text))
        return outs


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
        output_cols: str = "results",
        checkpoint_interval: int = 16,
        max_concurrency: int = 2,
        retries: int = 5,
        **extra_kwargs: Any,
    ):
        super().__init__(
            endpoint=endpoint,
            model=model,
            input_column_name=input_column_name,
            output_cols=output_cols,
            checkpoint_interval=checkpoint_interval,
            max_concurrency=max_concurrency,
            retries=retries,
            **extra_kwargs,
        )

    @deprecated(
        "Use transform_items instead for better performance with small batches."
    )
    async def transform(self, df: pd.DataFrame):
        # Copy input
        df = df.copy()

        await self.batched(
            df[self.input_column_name].tolist(),
            self._run_multi_turn,
        )

    async def _run_multi_turn(
        self, messages: List[ChatCompletionMessageParam]
    ) -> List[Dict[str, Any]]:
        # Find indices of user or tool messages
        user_idxs = [i for i, m in enumerate(messages) if m["role"] in {"user", "tool"}]
        if not user_idxs:
            raise ValueError("Input must contain at least one 'user' or 'tool' message")

        running: List[ChatCompletionMessageParam] = []

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

    async def transform_items(
        self, items: list[list[ChatCompletionMessageParam]]
    ) -> list[Any]:
        outs = []
        for msgs in items:
            # Find indices of user or tool messages
            user_idxs = [i for i, m in enumerate(msgs) if m["role"] in {"user", "tool"}]
            if not user_idxs:
                raise ValueError(
                    "Input must contain at least one 'user' or 'tool' message"
                )

            running: List[ChatCompletionMessageParam] = []

            # Zip together slice starts and ends
            idx = 0

            for i in user_idxs:
                # append all the messages until the next user message (including system, etc.)
                running.extend(msgs[idx : i + 1])

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
                    response_dict["reasoning_content"] = (
                        choice.message.reasoning_content
                    )
                running.append(response_dict)

                # update the index skipping assistant message
                idx = i + 2
            outs.append(running)
        return outs


class MultiTurnTranslationJob(SwarmJob):
    """
    For each row's `messages` (a list of dicts), replay the conversation
    turn by turn, replacing the messages with the assistant's translation.
    The translation system prompt is assumed to be the first message in the list
    and is prepended to each query.

    - Input  column: `messages`
    - Output column: `results`
    """

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        model: str = "",
        input_column_name: str = "messages",
        output_cols: str = "results",
        checkpoint_interval: int = 16,
        max_concurrency: int = 2,
        retries: int = 5,
        **extra_kwargs: Any,
    ):
        super().__init__(
            endpoint=endpoint,
            model=model,
            input_column_name=input_column_name,
            output_cols=output_cols,
            checkpoint_interval=checkpoint_interval,
            max_concurrency=max_concurrency,
            retries=retries,
            **extra_kwargs,
        )

    @deprecated(
        "Use transform_items instead for better performance with small batches."
    )
    async def transform(self, df: pd.DataFrame):
        # Copy input
        df = df.copy()

        await self.batched(
            df[self.input_column_name].tolist(),
            self._run_translation,
        )

    async def _run_translation(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        running: List[Dict[str, Any]] = []

        for i in range(1, len(messages)):
            # # skip assistant messages, only translate system/user/tool messages
            # if messages[i]["role"] == "assistant":
            #     continue

            query = [
                messages[0],
                {
                    "role": "user",
                    "content": messages[i]["content"],
                },
            ]

            resp: ChatCompletion = await self.client.chat.completions.create(
                model=self.model, messages=query, extra_body=self.kwargs
            )
            choice = resp.choices[0]

            # append the assistant's translation to the messages
            response_dict = {
                "role": messages[i]["role"],
                "content": choice.message.content,
            }
            if hasattr(choice.message, "reasoning_content"):
                response_dict["reasoning_content"] = choice.message.reasoning_content
            running.append(response_dict)

        return running

    async def transform_items(self, items: list[list[Dict[str, Any]]]) -> list[Any]:
        outs = []
        for msgs in items:
            running: List[Dict[str, Any]] = []

            for i in range(1, len(msgs)):
                # # skip assistant messages, only translate system/user/tool messages
                # if msgs[i]["role"] == "assistant":
                #     continue

                query = [
                    msgs[0],
                    {
                        "role": "user",
                        "content": msgs[i]["content"],
                    },
                ]

                resp: ChatCompletion = await self.client.chat.completions.create(
                    model=self.model, messages=query, extra_body=self.kwargs
                )
                choice = resp.choices[0]

                # append the assistant's translation to the messages
                response_dict = {
                    "role": msgs[i]["role"],
                    "content": choice.message.content,
                }
                if hasattr(choice.message, "reasoning_content"):
                    response_dict["reasoning_content"] = (
                        choice.message.reasoning_content
                    )
                running.append(response_dict)

            outs.append(running)
        return outs
