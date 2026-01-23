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

from collections.abc import Mapping
from typing import Any

from openai.types.chat import ChatCompletionAssistantMessageParam, ChatCompletionMessageParam
from openai.types.chat.chat_completion import ChatCompletion, Choice

from domyn_swarm.helpers.data import compute_perplexity_metrics, extract_token_logprobs
from domyn_swarm.jobs.api.base import SwarmJob


def _extract_reasoning_content(message: Any) -> Any | None:
    if message is None:
        return None
    if isinstance(message, Mapping):
        return message.get("reasoning_content")
    return getattr(message, "reasoning_content", None)


def _assistant_message_dict(*, role: str, content: Any, message: Any) -> dict[str, Any]:
    response_dict: dict[str, Any] = {"role": role, "content": content}
    reasoning_content = _extract_reasoning_content(message)
    if reasoning_content is not None:
        response_dict["reasoning_content"] = reasoning_content
    return response_dict


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

    async def transform_items(self, items: list[str]) -> list[Any]:
        """
        Transform a list of prompts into a list of completions.
        Each item in `items` is a single prompt string.

        Args:
            items: List of prompt strings.

        Returns a list of completion strings.
        """
        outs = []
        extra_body = self._request_kwargs()
        for prompt in items:
            resp = await self.client.completions.create(
                model=self.model, prompt=prompt, extra_body=extra_body
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
        self.output_cols = ["result", "reasoning_content"] if self.parse_reasoning else "result"

    async def transform_items(self, items: list[list[ChatCompletionMessageParam]]) -> list[Any]:
        outs = []
        extra_body = self._request_kwargs()
        for msgs in items:
            resp = await self.client.chat.completions.create(
                model=self.model, messages=msgs, extra_body=extra_body
            )
            choice = resp.choices[0]
            if self.parse_reasoning:
                outs.append(
                    (
                        choice.message.content,
                        _extract_reasoning_content(choice.message),
                    )
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

    async def transform_items(self, items: list[list[ChatCompletionMessageParam]]) -> list[Any]:
        outs = []
        extra_body = self._request_kwargs()
        for msgs in items:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=msgs,
                n=self.n,  # ask the API for n choices at once
                extra_body=extra_body,
            )
            outs.append([choice.message.content for choice in resp.choices])
        return outs


class PerplexityMixin:
    def compute_from_choice(self, choice: Choice) -> tuple[float, float]:
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

    async def transform_items(self, items: list[list[ChatCompletionMessageParam]]) -> list[Any]:
        outs = []
        extra_body = self._request_kwargs()
        for msgs in items:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=msgs,
                logprobs=True,
                extra_body=extra_body,
            )
            choice = resp.choices[0]
            text = choice.message.content
            perp, bottom50 = self.compute_from_choice(choice)
            outs.append((text, perp, bottom50))
        return outs


class MultiTurnChatCompletionJob(SwarmJob):
    """
    For each row's `messages` (a list of dicts), replay the conversation
    turn by turn, appending the assistant's reply after each user/tool message.

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

    async def _run_multi_turn(
        self,
        messages: list[ChatCompletionMessageParam],
        *,
        extra_body: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if extra_body is None:
            extra_body = self._request_kwargs()
        # Find indices of user or tool messages
        user_idxs = [i for i, m in enumerate(messages) if m["role"] in {"user", "tool"}]
        if not user_idxs:
            raise ValueError("Input must contain at least one 'user' or 'tool' message")

        running_request: list[ChatCompletionMessageParam] = []
        reasoning_by_index: list[Any | None] = []

        # Zip together slice starts and ends
        idx = 0

        for i in user_idxs:
            # append all the messages until the next user message (including system, etc.)
            running_request.extend(messages[idx : i + 1])
            reasoning_by_index.extend([None] * len(messages[idx : i + 1]))

            resp: ChatCompletion = await self.client.chat.completions.create(
                model=self.model, messages=running_request, extra_body=extra_body
            )
            choice = resp.choices[0]

            # append the assistant's response to the messages
            running_request.append(
                ChatCompletionAssistantMessageParam(
                    {"role": "assistant", "content": choice.message.content}
                )
            )
            reasoning_by_index.append(_extract_reasoning_content(choice.message))

            # update the index skipping assistant message
            idx = i + 2
        output: list[dict[str, Any]] = []
        for msg, reasoning in zip(running_request, reasoning_by_index, strict=True):
            msg_dict = dict(msg)
            if reasoning is not None and msg_dict.get("role") == "assistant":
                msg_dict["reasoning_content"] = reasoning
            output.append(msg_dict)
        return output

    async def transform_items(self, items: list[list[ChatCompletionMessageParam]]) -> list[Any]:
        outs = []
        extra_body = self._request_kwargs()

        outs = [await self._run_multi_turn(msgs, extra_body=extra_body) for msgs in items]
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

    async def _run_translation(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        extra_body = self._request_kwargs()
        running: list[dict[str, Any]] = []

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
                model=self.model, messages=query, extra_body=extra_body
            )
            choice = resp.choices[0]

            # append the assistant's translation to the messages
            running.append(
                _assistant_message_dict(
                    role=messages[i]["role"], content=choice.message.content, message=choice.message
                )
            )

        return running

    async def transform_items(self, items: list[list[dict[str, Any]]]) -> list[Any]:
        outs = []
        extra_body = self._request_kwargs()
        for msgs in items:
            running: list[dict[str, Any]] = []

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
                    model=self.model, messages=query, extra_body=extra_body
                )
                choice = resp.choices[0]

                # append the assistant's translation to the messages
                running.append(
                    _assistant_message_dict(
                        role=msgs[i]["role"], content=choice.message.content, message=choice.message
                    )
                )

            outs.append(running)
        return outs
