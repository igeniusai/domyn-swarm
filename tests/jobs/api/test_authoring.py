# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Writing a custom SwarmJob: the supported shape, and the errors along the way."""

import pytest

from domyn_swarm.jobs.api.base import SwarmJob
from domyn_swarm.jobs.api.config import JobConfig


@pytest.fixture(autouse=True)
def _endpoint(monkeypatch):
    monkeypatch.setenv("ENDPOINT", "http://dummy-endpoint")


class MyJobConfig(JobConfig):
    threshold: float = 0.5


class MyJob(SwarmJob):
    config_class = MyJobConfig
    config = MyJobConfig(input_column_name="prompt", output_cols="answer", max_concurrency=8)

    async def transform_items(self, items: list) -> list:
        return [f"{item}:{self.threshold}" for item in items]


def test_a_custom_job_needs_no_init() -> None:
    """The documented shape works with no constructor at all."""
    job = MyJob(model="gpt-4")
    assert job.input_column_name == "prompt"
    assert job.output_cols == "answer"
    assert job.max_concurrency == 8
    assert job.threshold == 0.5


def test_a_custom_field_is_overridable_and_reaches_transform() -> None:
    """Subclass config fields behave like any other."""
    job = MyJob(model="gpt-4", threshold=0.9)
    assert job.threshold == 0.9


def test_a_custom_field_is_not_sent_to_the_provider() -> None:
    """A subclass config field is configuration, not a request parameter."""
    assert MyJob(model="gpt-4", threshold=0.9)._request_kwargs() == {}


def test_a_custom_field_survives_the_round_trip() -> None:
    """Subclass configuration reaches the subprocess."""
    import json

    from domyn_swarm.jobs.api.builder import JobBuilder

    job = MyJob(model="gpt-4", threshold=0.9)
    payload = json.loads(JobBuilder.to_kwargs_json(job))
    assert payload["threshold"] == 0.9
    assert MyJob(model="gpt-4", **payload).threshold == 0.9


def test_a_config_of_the_wrong_type_is_rejected() -> None:
    """Passing a config the class cannot use says so."""
    with pytest.raises(TypeError, match="MyJobConfig"):
        MyJob(model="gpt-4", config=JobConfig(model="gpt-4"))


def test_a_typo_in_a_custom_field_is_rejected() -> None:
    """Near-miss detection covers subclass fields too, not just base ones."""
    with pytest.raises(TypeError, match="did you mean 'threshold'"):
        MyJob(model="gpt-4", threshhold=0.9)
