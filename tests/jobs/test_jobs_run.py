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

import sys
import tempfile
import types
from unittest.mock import patch

import pandas as pd
import pytest

from domyn_swarm.jobs import OutputJoinMode, SwarmJob
import domyn_swarm.jobs.run as run_mod
from domyn_swarm.jobs.run import (
    _amain,
    _load_cls,
    build_job_from_args,
    parse_args,
    run_job_unified,
)


# Dummy SwarmJob for testing
class DummySwarmJob(SwarmJob):
    max_concurrency = 2
    retries = 1
    timeout = 10
    output_mode = OutputJoinMode.APPEND

    def __init__(self, **kwargs):
        self.params = kwargs
        self.output_mode = kwargs.get("output_mode", OutputJoinMode.APPEND)

    async def transform(self, df):
        raise NotImplementedError()

    async def transform_items(self, items: list):
        return [f"test_shard_{i}" for i in items]


def test_load_cls_runtime():
    mod = types.ModuleType("fake_module")

    class Dummy:
        pass

    mod.Dummy = Dummy
    sys.modules["fake_module"] = mod

    loaded = _load_cls("fake_module:Dummy")
    assert loaded is Dummy


def test_build_job_from_args(monkeypatch):
    monkeypatch.setenv("JOB_CLASS", "domyn_swarm.jobs.run:DummySwarmJob")
    monkeypatch.setenv("MODEL", "gpt-4")
    monkeypatch.setenv("ENDPOINT", "http://localhost")

    monkeypatch.setattr(run_mod, "_load_cls", lambda path: DummySwarmJob)

    class Args:
        job_class = None
        model = None
        endpoint = None
        job_kwargs = '{"some_param": "value"}'

    job_cls, job_kwargs = build_job_from_args(Args())
    assert job_cls == DummySwarmJob
    assert job_kwargs["some_param"] == "value"
    assert job_kwargs["model"] == "gpt-4"
    assert job_kwargs["endpoint"] == "http://localhost"


@pytest.mark.asyncio
async def test_run_swarm_in_threads():
    df = pd.DataFrame({"messages": [1, 2, 3, 4]})
    out_df = await run_job_unified(
        DummySwarmJob,
        df,
        input_col="messages",
        output_cols=["output"],
        store_uri="file://" + tempfile.mkdtemp() + "/out.parquet",
        tag="test",
        nshards=2,
        checkpoint_dir=tempfile.mkdtemp(),
    )
    assert "output" in out_df.columns
    assert out_df.shape[0] == 4
    assert out_df["output"].str.startswith("test_shard").all()


def test_parse_args_minimal():
    args = parse_args(
        [
            "--job-class",
            "mod:Cls",
            "--model",
            "gpt-4",
            "--input-parquet",
            "in.parquet",
            "--output-parquet",
            "out.parquet",
            "--endpoint",
            "http://localhost",
        ]
    )
    assert args.model == "gpt-4"
    assert args.input_parquet.name == "in.parquet"


@pytest.mark.asyncio
async def test_amain_end_to_end(monkeypatch, tmp_path):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"

    # Prepare input dataframe
    df_in = pd.DataFrame({"text": ["hello", "world"]})
    df_in.to_parquet(input_path)

    # Patch helpers
    monkeypatch.setenv("JOB_CLASS", "domyn_swarm.jobs.run:DummySwarmJob")
    monkeypatch.setenv("MODEL", "mock-model")
    monkeypatch.setenv("ENDPOINT", "mock-endpoint")
    monkeypatch.setenv("INPUT_PARQUET", str(input_path))
    monkeypatch.setenv("OUTPUT_PARQUET", str(output_path))
    monkeypatch.setenv("JOB_KWARGS", '{"input_column_name": "text", "output_cols": "output"}')

    monkeypatch.setattr(run_mod, "_load_cls", lambda path: DummySwarmJob)

    args = [
        "--nthreads",
        "1",  # Use single-threaded path
        "--job-kwargs",
        '{"input_column_name": "text", "output_cols": "output"}',
    ]
    await _amain(args)

    df_out = pd.read_parquet(output_path)
    print(df_out)
    assert "output" in df_out.columns
    assert df_out.shape[0] == 2


def test_main_wrapper(monkeypatch):
    # Use parse_args via monkeypatch to avoid invoking asyncio in this test
    with patch("domyn_swarm.jobs.run._amain") as mock_amain:
        from domyn_swarm.jobs.run import main

        main(
            [
                "--job-class",
                "mod:Cls",
                "--model",
                "gpt",
                "--input-parquet",
                "in.pq",
                "--output-parquet",
                "out.pq",
                "--endpoint",
                "ep",
            ]
        )
        mock_amain.assert_called_once()
