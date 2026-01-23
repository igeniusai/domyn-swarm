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

"""Compatibility shim for deprecated imports."""

from __future__ import annotations

import warnings

from domyn_swarm.jobs.api.chat_completion import (
    ChatCompletionJob,
    ChatCompletionPerplexityJob,
    CompletionJob,
    MultiChatCompletionJob,
    MultiTurnChatCompletionJob,
    MultiTurnTranslationJob,
    PerplexityMixin,
    _assistant_message_dict,
    _extract_reasoning_content,
    compute_perplexity_metrics,
    extract_token_logprobs,
)

warnings.warn(
    "domyn_swarm.jobs.chat_completion is deprecated; use domyn_swarm.jobs.api.chat_completion",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "ChatCompletionJob",
    "ChatCompletionPerplexityJob",
    "CompletionJob",
    "MultiChatCompletionJob",
    "MultiTurnChatCompletionJob",
    "MultiTurnTranslationJob",
    "PerplexityMixin",
    "_assistant_message_dict",
    "_extract_reasoning_content",
    "compute_perplexity_metrics",
    "extract_token_logprobs",
]
