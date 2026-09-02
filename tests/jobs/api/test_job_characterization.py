# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""The construction and serialization behaviour `SwarmJob` guarantees today.

Pins the base constructor's effective defaults, each shipped job class's
resolved output columns, and the serialize/reconstruct round trip that carries
a job to the cluster. A failure here is a behaviour change.
"""

import json

import pytest

from domyn_swarm.jobs.api.base import OutputJoinMode, SwarmJob
from domyn_swarm.jobs.api.builder import JobBuilder
from domyn_swarm.jobs.api.chat_completion import (
    ChatCompletionJob,
    CompletionJob,
    MultiChatCompletionJob,
    MultiTurnChatCompletionJob,
    MultiTurnTranslationJob,
)


@pytest.fixture(autouse=True)
def _endpoint(monkeypatch):
    monkeypatch.setenv("ENDPOINT", "http://dummy-endpoint")


class Plain(SwarmJob):
    async def transform_items(self, items: list):
        return items


def test_base_defaults() -> None:
    """The base constructor's effective defaults."""
    job = Plain(model="m")
    assert job.name == "Plain"
    assert job.model == "m"
    assert job.provider == "openai"
    assert job.input_column_name == "messages"
    assert job.id_column_name is None
    assert job.output_cols == "result"
    assert job.default_output_cols == ["result"]
    assert job.output_mode is OutputJoinMode.APPEND
    assert job.checkpoint_interval == 16
    assert job.max_concurrency == 2
    assert job.retries == 5
    assert job.timeout == 600
    assert job.data_backend is None
    assert job.native_backend is False
    assert job.native_batch_size is None


def test_default_output_cols_is_derived_from_output_cols() -> None:
    """A string `output_cols` becomes a one-element `default_output_cols`."""
    assert Plain(model="m", output_cols="answer").default_output_cols == ["answer"]
    assert Plain(model="m", output_cols=["a", "b"]).default_output_cols == ["a", "b"]


def test_explicit_default_output_cols_wins() -> None:
    """An explicit `default_output_cols` is not overwritten by the derivation."""
    job = Plain(model="m", output_cols="answer", default_output_cols=["x"])
    assert job.default_output_cols == ["x"]


def test_missing_endpoint_raises(monkeypatch) -> None:
    """No endpoint and no ENDPOINT env var is a RuntimeError."""
    monkeypatch.delenv("ENDPOINT", raising=False)
    with pytest.raises(RuntimeError, match="ENDPOINT"):
        Plain(model="m")


def test_missing_model_raises() -> None:
    """An empty model name is a ValueError."""
    with pytest.raises(ValueError, match="Model name"):
        Plain()


def test_provider_params_reach_request_kwargs() -> None:
    """Unrecognised kwargs are forwarded as provider request parameters."""
    job = Plain(model="m", temperature=0.2, top_p=0.9)
    assert job._request_kwargs() == {"temperature": 0.2, "top_p": 0.9}


@pytest.mark.parametrize(
    "factory,expected_output_cols",
    [
        (lambda: CompletionJob(model="m"), "completion"),
        (lambda: ChatCompletionJob(model="m"), "result"),
        (
            lambda: ChatCompletionJob(model="m", parse_reasoning=True),
            ["result", "reasoning_content"],
        ),
        (
            lambda: MultiChatCompletionJob(model="m", output_cols="gen", n=3),
            ["gen_1", "gen_2", "gen_3"],
        ),
        (lambda: MultiTurnChatCompletionJob(model="m"), "results"),
        (lambda: MultiTurnTranslationJob(model="m"), "results"),
    ],
)
def test_shipped_jobs_output_cols(factory, expected_output_cols) -> None:
    """Each shipped job's resolved output columns."""
    assert factory().output_cols == expected_output_cols


def test_multi_chat_without_output_cols_raises() -> None:
    """MultiChatCompletionJob cannot be built without an explicit output_cols.

    A defect rather than intended behaviour: `__init__` calls
    `kwargs.pop("output_cols")` with no default.
    """
    with pytest.raises(KeyError, match="output_cols"):
        MultiChatCompletionJob(model="m")


@pytest.mark.parametrize(
    "factory",
    [
        lambda: CompletionJob(model="m"),
        lambda: ChatCompletionJob(model="m"),
        lambda: ChatCompletionJob(model="m", parse_reasoning=True),
        lambda: MultiChatCompletionJob(model="m", output_cols="gen", n=2),
        lambda: MultiTurnChatCompletionJob(model="m"),
        lambda: MultiTurnTranslationJob(model="m"),
        lambda: Plain(model="m", temperature=0.2),
    ],
)
def test_serialize_reconstruct_round_trip(factory) -> None:
    """A job survives the trip through `--job-kwargs` and back.

    This is how every job reaches the cluster: the payload is serialized here,
    passed on the command line, and used to rebuild the job in the subprocess.
    """
    job = factory()
    payload = json.loads(JobBuilder.to_kwargs_json(job))
    rebuilt = type(job)(model=job.model, **payload)

    assert rebuilt.output_cols == job.output_cols
    assert rebuilt.default_output_cols == job.default_output_cols
    assert rebuilt.input_column_name == job.input_column_name
    assert rebuilt.max_concurrency == job.max_concurrency
    assert rebuilt.retries == job.retries
    assert rebuilt.timeout == job.timeout
    assert rebuilt.output_mode == job.output_mode
    assert rebuilt._request_kwargs() == job._request_kwargs()


def test_output_column_name_deprecation_path() -> None:
    """The deprecated `output_column_name` parameter still sets `output_cols`."""
    with pytest.warns(DeprecationWarning, match="output_column_name.*deprecated"):
        job = Plain(model="m", output_column_name="legacy")
    assert job.output_cols == "legacy"
