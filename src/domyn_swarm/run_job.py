# domyn_swarm/run_job.py  (installed with your library)
import os, json, importlib, pandas as pd, asyncio
from domyn_swarm.jobs import SwarmJob  # base class


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

    df_in = pd.read_parquet(in_path)
    df_out = await job.transform(df_in)  # async core
    df_out.to_parquet(out_path)


def main():
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
