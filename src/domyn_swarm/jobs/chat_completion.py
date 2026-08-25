# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

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
