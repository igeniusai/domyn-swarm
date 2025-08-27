import asyncio
from typing import Any, Callable, List, Optional

import numpy as np
import pandas as pd

from domyn_swarm.checkpoint.store import ParquetShardStore
from domyn_swarm.jobs.base import SwarmJob
from domyn_swarm.jobs.runner import JobRunner, RunnerConfig


def _has_new_api(job: SwarmJob) -> bool:
    return hasattr(job, "transform_streaming") and callable(job.transform_streaming)


def _has_old_api(job: SwarmJob) -> bool:
    return hasattr(job, "run") and callable(job.run)


async def run_job_unified(
    job_factory: Callable[[], Any],
    df: pd.DataFrame,
    *,
    input_col: str,
    output_cols: Optional[List[str]],
    nshards: int = 1,
    store_uri: Optional[str] = None,
    checkpoint_every: int = 16,
    tag: Optional[str] = None,
    checkpoint_dir: Optional[str] = ".checkpoints",
) -> pd.DataFrame:
    job_probe = job_factory()

    # New-style path (store-agnostic, streaming flush)
    if _has_new_api(job_probe):
        if store_uri is None:
            raise ValueError("store_uri is required for new-style jobs.")
        cfg = RunnerConfig(id_col="_row_id", checkpoint_every=checkpoint_every)
        if nshards <= 1:
            store = ParquetShardStore(store_uri)
            return await JobRunner(store, cfg).run(
                job_probe, df, input_col=input_col, output_cols=output_cols
            )
        indices = np.array_split(df.index, nshards)

        async def _one(i, idx):
            sub = df.loc[idx].copy()
            su = store_uri.replace(".parquet", f"_shard{i}.parquet")
            store = ParquetShardStore(su)
            return await JobRunner(store, cfg).run(
                job_factory(), sub, input_col=input_col, output_cols=output_cols
            )

        parts = await asyncio.gather(*[_one(i, idx) for i, idx in enumerate(indices)])
        return pd.concat(parts).sort_index()

    # Old-style path (full BC)
    if not _has_old_api(job_probe):
        raise TypeError(
            "Job must implement either the new streaming API or legacy run(df, ...) API"
        )

    if nshards <= 1:
        return await job_probe.run(df, tag=tag or "run", checkpoint_dir=checkpoint_dir)
    indices = np.array_split(df.index, nshards)

    async def _old_one(i, idx):
        sub = df.loc[idx].copy()
        return await job_factory().run(
            sub, tag=f"{tag or 'run'}_shard{i}", checkpoint_dir=checkpoint_dir
        )

    parts = await asyncio.gather(*[_old_one(i, idx) for i, idx in enumerate(indices)])
    return pd.concat(parts).sort_index()
