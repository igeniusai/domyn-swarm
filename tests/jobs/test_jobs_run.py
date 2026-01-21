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


class DummySwarmJob(SwarmJob):
    max_concurrency = 2
    retries = 1
    timeout = 10
    output_mode = OutputJoinMode.APPEND

    def __init__(self, **kwargs):
        kwargs.setdefault("endpoint", "http://dummy-endpoint")
        kwargs.setdefault("model", "dummy-model")
        kwargs.setdefault("input_column_name", "messages")
        kwargs.setdefault("output_cols", "output")
        super().__init__(**kwargs)
        self.params = kwargs
        self.output_mode = kwargs.get("output_mode", OutputJoinMode.APPEND)

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
async def test_run_job_unified_streaming():
    df = pd.DataFrame({"messages": [1, 2, 3, 4]})
    out_df = await run_job_unified(
        DummySwarmJob,
        df,
        input_col="messages",
        output_cols=["output"],
        store_uri="file://" + tempfile.mkdtemp() + "/out.parquet",
        nshards=2,
    )
    assert "output" in out_df.columns
    assert out_df.shape[0] == 4
    assert out_df["output"].str.startswith("test_shard").all()


@pytest.mark.asyncio
async def test_run_job_unified_requires_id_column(tmp_path):
    df = pd.DataFrame({"doc_id": [10, 11], "messages": [1, 2]})
    out_df = await run_job_unified(
        lambda: DummySwarmJob(id_column_name="doc_id"),
        df,
        input_col="messages",
        output_cols=["output"],
        store_uri=f"file://{tmp_path / 'out.parquet'}",
    )
    assert out_df["doc_id"].tolist() == [10, 11]
    assert out_df["output"].str.startswith("test_shard").all()


@pytest.mark.asyncio
async def test_run_job_unified_missing_id_column_raises(tmp_path):
    df = pd.DataFrame({"messages": [1, 2]})
    with pytest.raises(ValueError, match="id column"):
        await run_job_unified(
            lambda: DummySwarmJob(id_column_name="doc_id"),
            df,
            input_col="messages",
            output_cols=["output"],
            store_uri=f"file://{tmp_path / 'out.parquet'}",
        )


@pytest.mark.asyncio
async def test_run_job_unified_arrow_runner_id_column(tmp_path):
    df = pd.DataFrame({"doc_id": [10, 11], "messages": [1, 2]})
    out_df = await run_job_unified(
        lambda: DummySwarmJob(id_column_name="doc_id"),
        df,
        input_col="messages",
        output_cols=["output"],
        store_uri=f"file://{tmp_path / 'out.parquet'}",
        runner="arrow",
    )
    assert out_df["doc_id"].tolist() == [10, 11]
    assert out_df["output"].tolist() == ["test_shard_1", "test_shard_2"]


@pytest.mark.asyncio
async def test_run_job_unified_polars_runner_lazy(tmp_path):
    """Run the Arrow-backed polars path with LazyFrame input.

    Args:
        tmp_path: Pytest temporary directory.
    """
    pl = pytest.importorskip("polars")
    data = pl.DataFrame({"doc_id": [10, 11], "messages": [1, 2]}).lazy()
    out_df = await run_job_unified(
        lambda: DummySwarmJob(id_column_name="doc_id"),
        data,
        input_col="messages",
        output_cols=["output"],
        store_uri=f"file://{tmp_path / 'out.parquet'}",
        data_backend="polars",
        runner="arrow",
    )
    assert isinstance(out_df, pl.DataFrame)
    assert out_df["doc_id"].to_list() == [10, 11]
    assert out_df["output"].to_list() == ["test_shard_1", "test_shard_2"]


@pytest.mark.asyncio
async def test_arrow_runner_output_modes_pandas(tmp_path):
    """Validate Arrow runner output modes for pandas backend.

    Args:
        tmp_path: Pytest temporary directory.
    """
    df = pd.DataFrame({"messages": [1, 2], "extra": ["a", "b"]})
    expected_cols = {
        OutputJoinMode.APPEND: {"_row_id", "messages", "extra", "output"},
        OutputJoinMode.IO_ONLY: {"_row_id", "messages", "output"},
        OutputJoinMode.REPLACE: {"_row_id", "output"},
    }
    for mode, cols in expected_cols.items():
        out_df = await run_job_unified(
            lambda m=mode: DummySwarmJob(output_mode=m),
            df,
            input_col="messages",
            output_cols=["output"],
            store_uri=f"file://{tmp_path / f'out_{mode.value}.parquet'}",
            runner="arrow",
        )
        assert set(out_df.columns) == cols


@pytest.mark.asyncio
async def test_arrow_runner_output_modes_polars(tmp_path):
    """Validate Arrow runner output modes for polars backend.

    Args:
        tmp_path: Pytest temporary directory.
    """
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"messages": [1, 2], "extra": ["a", "b"]})
    expected_cols = {
        OutputJoinMode.APPEND: {"_row_id", "messages", "extra", "output"},
        OutputJoinMode.IO_ONLY: {"_row_id", "messages", "output"},
        OutputJoinMode.REPLACE: {"_row_id", "output"},
    }
    for mode, cols in expected_cols.items():
        out_df = await run_job_unified(
            lambda m=mode: DummySwarmJob(output_mode=m),
            df,
            input_col="messages",
            output_cols=["output"],
            store_uri=f"file://{tmp_path / f'out_{mode.value}.parquet'}",
            data_backend="polars",
            runner="arrow",
        )
        assert isinstance(out_df, pl.DataFrame)
        assert set(out_df.columns) == cols


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
async def test_transform_is_not_supported():
    job = DummySwarmJob()
    with pytest.raises(RuntimeError, match="transform\\(df\\) is no longer supported"):
        await job.transform(pd.DataFrame({"messages": ["hi"]}))


@pytest.mark.asyncio
async def test_amain_end_to_end(monkeypatch, tmp_path):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    checkpoint_dir = tmp_path / "checkpoints"

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
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--job-kwargs",
        '{"input_column_name": "text", "output_cols": "output"}',
    ]
    await _amain(args)

    df_out = pd.read_parquet(output_path)
    print(df_out)
    assert "output" in df_out.columns
    assert df_out.shape[0] == 2


@pytest.mark.asyncio
async def test_amain_end_to_end_polars_backend(monkeypatch, tmp_path):
    pytest.importorskip("polars")

    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    checkpoint_dir = tmp_path / "checkpoints"

    df_in = pd.DataFrame({"text": ["hello", "world"]})
    df_in.to_parquet(input_path)

    monkeypatch.setenv("JOB_CLASS", "domyn_swarm.jobs.run:DummySwarmJob")
    monkeypatch.setenv("MODEL", "mock-model")
    monkeypatch.setenv("ENDPOINT", "mock-endpoint")
    monkeypatch.setenv("INPUT_PARQUET", str(input_path))
    monkeypatch.setenv("OUTPUT_PARQUET", str(output_path))
    monkeypatch.setenv(
        "JOB_KWARGS",
        '{"input_column_name":"text","output_cols":"output","data_backend":"polars"}',
    )

    monkeypatch.setattr(run_mod, "_load_cls", lambda path: DummySwarmJob)

    args = [
        "--nthreads",
        "1",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--job-kwargs",
        '{"input_column_name":"text","output_cols":"output","data_backend":"polars"}',
    ]
    await _amain(args)

    df_out = pd.read_parquet(output_path)
    assert "output" in df_out.columns
    assert df_out.shape[0] == 2


@pytest.mark.asyncio
async def test_amain_arrow_runner_pandas_backend(monkeypatch, tmp_path):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    checkpoint_dir = tmp_path / "checkpoints"

    df_in = pd.DataFrame({"text": ["hello", "world"]})
    df_in.to_parquet(input_path)

    monkeypatch.setenv("JOB_CLASS", "domyn_swarm.jobs.run:DummySwarmJob")
    monkeypatch.setenv("MODEL", "mock-model")
    monkeypatch.setenv("ENDPOINT", "mock-endpoint")
    monkeypatch.setenv("INPUT_PARQUET", str(input_path))
    monkeypatch.setenv("OUTPUT_PARQUET", str(output_path))
    monkeypatch.setenv(
        "JOB_KWARGS",
        '{"input_column_name":"text","output_cols":"output","data_backend":"pandas"}',
    )

    monkeypatch.setattr(run_mod, "_load_cls", lambda path: DummySwarmJob)

    args = [
        "--nthreads",
        "1",
        "--runner",
        "arrow",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--job-kwargs",
        '{"input_column_name":"text","output_cols":"output","data_backend":"pandas"}',
    ]
    await _amain(args)

    df_out = pd.read_parquet(output_path)
    assert "output" in df_out.columns
    assert df_out.shape[0] == 2


@pytest.mark.asyncio
async def test_amain_arrow_runner_sharded(monkeypatch, tmp_path):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    checkpoint_dir = tmp_path / "checkpoints"

    df_in = pd.DataFrame({"text": ["hello", "world"]})
    df_in.to_parquet(input_path)

    monkeypatch.setenv("JOB_CLASS", "domyn_swarm.jobs.run:DummySwarmJob")
    monkeypatch.setenv("MODEL", "mock-model")
    monkeypatch.setenv("ENDPOINT", "mock-endpoint")
    monkeypatch.setenv("INPUT_PARQUET", str(input_path))
    monkeypatch.setenv("OUTPUT_PARQUET", str(output_path))
    monkeypatch.setenv(
        "JOB_KWARGS",
        '{"input_column_name":"text","output_cols":"output","data_backend":"pandas"}',
    )

    monkeypatch.setattr(run_mod, "_load_cls", lambda path: DummySwarmJob)

    args = [
        "--nthreads",
        "2",
        "--runner",
        "arrow",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--job-kwargs",
        '{"input_column_name":"text","output_cols":"output","data_backend":"pandas"}',
    ]
    await _amain(args)

    df_out = pd.read_parquet(output_path)
    assert "output" in df_out.columns
    assert df_out.shape[0] == 2


@pytest.mark.asyncio
async def test_amain_no_resume_forces_recompute(monkeypatch, tmp_path):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    checkpoint_dir = tmp_path / "checkpoints"

    df_in = pd.DataFrame({"text": ["hello", "world"]})
    df_in.to_parquet(input_path)

    monkeypatch.setenv("JOB_CLASS", "domyn_swarm.jobs.run:DummySwarmJob")
    monkeypatch.setenv("MODEL", "mock-model")
    monkeypatch.setenv("ENDPOINT", "mock-endpoint")
    monkeypatch.setenv("INPUT_PARQUET", str(input_path))
    monkeypatch.setenv("OUTPUT_PARQUET", str(output_path))
    monkeypatch.setenv("JOB_KWARGS", '{"input_column_name": "text", "output_cols": "output"}')
    monkeypatch.setattr(run_mod, "_load_cls", lambda path: DummySwarmJob)

    # First run creates checkpoints
    await _amain(
        [
            "--nthreads",
            "1",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--job-kwargs",
            '{"input_column_name": "text", "output_cols": "output"}',
        ]
    )

    # Second run should resume and skip work even if transform_items would raise.
    async def _raise_transform_items(self, items: list):
        raise RuntimeError("should not be called when resuming")

    monkeypatch.setattr(DummySwarmJob, "transform_items", _raise_transform_items)
    await _amain(
        [
            "--nthreads",
            "1",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--job-kwargs",
            '{"input_column_name": "text", "output_cols": "output"}',
        ]
    )

    # With --no-resume, we should recompute and therefore hit the raising transform_items.
    with pytest.raises(RuntimeError, match="should not be called"):
        await _amain(
            [
                "--nthreads",
                "1",
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--no-resume",
                "--job-kwargs",
                '{"input_column_name": "text", "output_cols": "output"}',
            ]
        )


@pytest.mark.asyncio
async def test_amain_no_checkpointing_forces_recompute(monkeypatch, tmp_path):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    checkpoint_dir = tmp_path / "checkpoints"

    df_in = pd.DataFrame({"text": ["hello", "world"]})
    df_in.to_parquet(input_path)

    monkeypatch.setenv("JOB_CLASS", "domyn_swarm.jobs.run:DummySwarmJob")
    monkeypatch.setenv("MODEL", "mock-model")
    monkeypatch.setenv("ENDPOINT", "mock-endpoint")
    monkeypatch.setenv("INPUT_PARQUET", str(input_path))
    monkeypatch.setenv("OUTPUT_PARQUET", str(output_path))
    monkeypatch.setenv("JOB_KWARGS", '{"input_column_name": "text", "output_cols": "output"}')
    monkeypatch.setattr(run_mod, "_load_cls", lambda path: DummySwarmJob)

    # First run creates checkpoints
    await _amain(
        [
            "--nthreads",
            "1",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--job-kwargs",
            '{"input_column_name": "text", "output_cols": "output"}',
        ]
    )

    async def _raise_transform_items(self, items: list):
        raise RuntimeError("must be called when checkpointing is disabled")

    monkeypatch.setattr(DummySwarmJob, "transform_items", _raise_transform_items)
    with pytest.raises(RuntimeError, match="checkpointing is disabled"):
        await _amain(
            [
                "--nthreads",
                "1",
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--no-checkpointing",
                "--job-kwargs",
                '{"input_column_name": "text", "output_cols": "output"}',
            ]
        )


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
