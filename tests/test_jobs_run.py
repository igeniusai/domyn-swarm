import sys
import types
import pytest
import pandas as pd
import tempfile
import domyn_swarm.jobs.run as run_mod
from unittest.mock import patch

from domyn_swarm.jobs.run import (
    _load_cls,
    build_job_from_args,
    run_swarm_in_threads,
    parse_args,
    _amain,
)


# Dummy SwarmJob for testing
class DummySwarmJob:
    def __init__(self, **kwargs):
        self.params = kwargs

    async def run(self, df: pd.DataFrame, tag: str, checkpoint_dir: str = "."):
        df = df.copy()
        df["output"] = f"{tag}_processed"
        return df


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
    df = pd.DataFrame({"col": [1, 2, 3, 4]})
    out_df = run_swarm_in_threads(
        df,
        DummySwarmJob,
        job_kwargs={},
        tag="test",
        num_threads=2,
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

    monkeypatch.setattr(run_mod, "_load_cls", lambda path: DummySwarmJob)

    args = [
        "--nthreads",
        "1",  # Use single-threaded path
    ]
    await _amain(args)

    df_out = pd.read_parquet(output_path)
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
