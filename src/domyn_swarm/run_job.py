# domyn_swarm/run_job.py  (installed with your library)
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


async def _amain() -> None:
    cls_path = os.environ["JOB_CLASS"]  # "pkg.module:Class"
    model = os.environ["MODEL"]
    in_path = os.environ["INPUT_PARQUET"]
    out_path = os.environ["OUTPUT_PARQUET"]
    kwargs = json.loads(os.getenv("JOB_KWARGS", "{}"))
    endpoint = os.environ["ENDPOINT"]

    JobCls = _load_cls(cls_path)
    job = JobCls(endpoint=endpoint, model=model, **kwargs)

    tag = parquet_hash(pathlib.Path(in_path))
    df_in = pd.read_parquet(in_path)
    df_out: pd.DataFrame = await job.run(df_in, tag)
    df_out.to_parquet(out_path)


def main():
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
