# domyn_swarm/run_job.py  (installed with your library)
import argparse
import os
import json
import importlib
import pandas as pd
import asyncio
from domyn_swarm.helpers import parquet_hash
from domyn_swarm.jobs import SwarmJob  # base class
import pathlib


def _load_cls(path: str) -> type[SwarmJob]:
    mod, cls = path.split(":")
    return getattr(importlib.import_module(mod), cls)


def parse_args():
    parser = argparse.ArgumentParser(description="Run a SwarmJob on a Parquet input.")

    parser.add_argument(
        "--job-class", type=str, help="Job class path in format 'pkg.module:Class'"
    )
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument(
        "--input-parquet", type=pathlib.Path, help="Path to input Parquet file"
    )
    parser.add_argument(
        "--output-parquet", type=pathlib.Path, help="Path to output Parquet file"
    )
    parser.add_argument("--endpoint", type=str, help="Endpoint URL")
    parser.add_argument(
        "--job-kwargs", type=str, default="{}", help="Extra JSON string kwargs"
    )

    return parser.parse_args()


async def _amain():
    args = parse_args()

    cls_path = args.job_class or os.environ["JOB_CLASS"]
    model = args.model or os.environ["MODEL"]
    in_path = args.input_parquet or pathlib.Path(os.environ["INPUT_PARQUET"])
    out_path = args.output_parquet or pathlib.Path(os.environ["OUTPUT_PARQUET"])
    endpoint = args.endpoint or os.environ["ENDPOINT"]

    job_params = json.loads(args.job_kwargs or os.getenv("JOB_KWARGS", "{}"))
    kwargs = job_params.pop("kwargs", {})

    JobCls = _load_cls(cls_path)
    job = JobCls(endpoint=endpoint, model=model, **job_params, **kwargs)

    tag = parquet_hash(in_path)
    df_in = pd.read_parquet(in_path)
    df_out: pd.DataFrame = await job.run(df_in, tag)
    os.makedirs(out_path.parent, exist_ok=True)
    df_out.to_parquet(out_path)


def main():
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
