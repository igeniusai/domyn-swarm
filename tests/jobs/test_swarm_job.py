# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from domyn_swarm.jobs.api.base import SwarmJob


class DummyJob(SwarmJob):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []

    async def transform_items(self, items):
        return [await self.fake_fn(item) for item in items]

    async def fake_fn(self, x):
        self.calls.append(x)
        return f"out_{x}"


@pytest.mark.asyncio
async def test_swarm_job_checkpointing(tmp_path):
    df = pd.DataFrame({"messages": ["hi", "yo", "hello"]})
    job = DummyJob(endpoint="http://localhost", model="fake", checkpoint_dir=str(tmp_path))

    result = await job.run(df, tag="test", checkpoint_dir=tmp_path)
    assert set(result.columns) >= {"messages", "result"}
    assert job.calls == ["hi", "yo", "hello"]


# ---------------------------
# Constructor typo detection
# ---------------------------


def test_misspelled_parameter_is_rejected_with_a_suggestion():
    """A near-miss of a real parameter is a typo, not a request param."""
    with pytest.raises(TypeError) as excinfo:
        DummyJob(endpoint="http://localhost", model="fake", max_concurency=999)

    message = str(excinfo.value)
    assert "max_concurency" in message
    assert "max_concurrency" in message


def test_misspelled_parameter_is_caught_before_it_reaches_a_request():
    """The typo must not survive into the request body."""
    with pytest.raises(TypeError):
        DummyJob(endpoint="http://localhost", model="fake", chekpoint_interval=4)


def test_genuine_request_parameters_still_pass_through():
    """`--job-kwargs '{"temperature":0.2}'` is a documented feature; keep it working."""
    job = DummyJob(
        endpoint="http://localhost",
        model="fake",
        temperature=0.2,
        top_p=0.9,
    )

    assert job.kwargs["temperature"] == 0.2
    assert job._request_kwargs() == {"temperature": 0.2, "top_p": 0.9}


def test_correctly_spelled_parameters_are_not_flagged():
    """An exact parameter name is consumed normally, never treated as a typo."""
    job = DummyJob(endpoint="http://localhost", model="fake", max_concurrency=8)

    assert job.max_concurrency == 8
    assert "max_concurrency" not in job.kwargs
