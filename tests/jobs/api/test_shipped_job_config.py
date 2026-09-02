# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Per-class configuration on the shipped job classes.

`MultiChatCompletionJob.n` and `ChatCompletionJob.parse_reasoning` are
configuration fields, not instance state: they must survive the trip through
`--job-kwargs`, or a job rebuilt on a cluster node falls back to the field
default instead of the value its caller chose. `parse_reasoning` must also
stay out of the provider request parameters.
"""

import json

import pytest

from domyn_swarm.jobs.api.builder import JobBuilder
from domyn_swarm.jobs.api.chat_completion import ChatCompletionJob, MultiChatCompletionJob


@pytest.fixture(autouse=True)
def _endpoint(monkeypatch):
    monkeypatch.setenv("ENDPOINT", "http://dummy-endpoint")


def test_multi_chat_completion_n_survives_the_wire() -> None:
    """A non-default `n` is not silently dropped on rebuild.

    Were it missing from the payload, the rebuilt job would fall back to `n=3`
    and ask the provider for the wrong number of completions.
    """
    job = MultiChatCompletionJob(model="m", output_cols="gen", n=5)
    payload = json.loads(JobBuilder.to_kwargs_json(job))

    assert payload["n"] == 5

    rebuilt = type(job)(model=job.model, **payload)
    assert rebuilt.n == job.n == 5


def test_chat_completion_parse_reasoning_survives_the_wire() -> None:
    """A non-default `parse_reasoning` is not silently dropped on rebuild."""
    job = ChatCompletionJob(model="m", parse_reasoning=True)
    payload = json.loads(JobBuilder.to_kwargs_json(job))

    assert payload["parse_reasoning"] is True

    rebuilt = type(job)(model=job.model, **payload)
    assert rebuilt.parse_reasoning == job.parse_reasoning is True
    assert rebuilt.output_cols == job.output_cols == ["result", "reasoning_content"]


def test_chat_completion_parse_reasoning_does_not_reach_the_provider() -> None:
    """`parse_reasoning` is job configuration, not a provider request parameter."""
    job = ChatCompletionJob(model="m", parse_reasoning=True)
    assert "parse_reasoning" not in job._request_kwargs()
