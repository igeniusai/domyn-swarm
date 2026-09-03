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
async def test_swarm_job_checkpointing(tmp_path, recwarn):
    """`run()` takes its own `checkpoint_dir`; the constructor has no such field.

    Passing one to the constructor is not an error -- it becomes a
    `request_params` entry sent to the provider -- so the assertions below
    check that nothing was synthesized there.
    """
    df = pd.DataFrame({"messages": ["hi", "yo", "hello"]})
    job = DummyJob(endpoint="http://localhost", model="fake")

    result = await job.run(df, tag="test", checkpoint_dir=tmp_path)
    assert set(result.columns) >= {"messages", "result"}
    assert job.calls == ["hi", "yo", "hello"]
    assert job._request_kwargs() == {}, "no provider parameter should have been synthesized"
    assert [w for w in recwarn if issubclass(w.category, DeprecationWarning)] == []


# ---------------------------
# Constructor typo detection
# ---------------------------


def test_misspelled_parameter_is_rejected_with_a_suggestion():
    """A near-miss of a configuration field is a typo, not a request parameter.

    Routing it into `request_params` would leave `max_concurrency` at its
    default while sending `max_concurency` to the provider.
    """
    with pytest.raises(TypeError, match="did you mean 'max_concurrency'") as excinfo:
        DummyJob(endpoint="http://localhost", model="fake", max_concurency=999)

    message = str(excinfo.value)
    assert "max_concurency" in message
    assert "max_concurrency" in message


def test_misspelled_parameter_is_caught_before_it_reaches_a_request():
    """The typo must not survive into the request body."""
    with pytest.raises(TypeError, match="did you mean 'checkpoint_interval'"):
        DummyJob(endpoint="http://localhost", model="fake", chekpoint_interval=4)


def test_genuine_request_parameters_still_pass_through():
    """`--job-kwargs '{"temperature":0.2}'` is a documented feature; keep it working.

    Bare constructor kwargs are deprecated in favour of `request_params={...}`,
    but still route into it rather than being rejected.
    """
    with pytest.warns(DeprecationWarning, match="request_params"):
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
