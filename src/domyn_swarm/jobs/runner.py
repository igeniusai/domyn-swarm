from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from domyn_swarm.checkpoint.store import CheckpointStore, FlushBatch, ParquetShardStore
from domyn_swarm.jobs.base import SwarmJob


@dataclass
class RunnerConfig:
    id_col: str = "_row_id"
    checkpoint_every: int = 16
    total_concurrency: Optional[int] = None  # reserved for future admission control


class JobRunner:
    """Unifies single- and multi-shard execution and decouples checkpointing.

    The job object is expected to expose a **streaming** transform:

        await job.transform_streaming(items: list[Any], *,
                                      on_flush: Callable[[list[int], list[Any]], Awaitable[None]],
                                      checkpoint_every: int) -> None

    If your current jobs implement `transform(df)` + `batched(...)`, create a thin
    adapter method on the job to satisfy `transform_streaming`.

    Parameters
    ----------
    store : CheckpointStore
        The persistence backend (local or cloud).
    cfg : RunnerConfig
        Runner tuning parameters.
    input_col : str
        Column name to read inputs from (e.g., "messages").
    output_cols : list[str] | None
        Output columns (None means the job returns dicts that will be normalized).
    """

    def __init__(self, store: CheckpointStore, cfg: RunnerConfig | None = None):
        self.store = store
        self.cfg = cfg or RunnerConfig()

    async def run(
        self,
        job: SwarmJob,
        df: pd.DataFrame,
        *,
        input_col: str,
        output_cols: Optional[list[str]],
    ) -> pd.DataFrame:
        df = df.copy()
        if self.cfg.id_col not in df.columns:
            df[self.cfg.id_col] = df.index
        todo = self.store.prepare(df, self.cfg.id_col)

        ids = todo[self.cfg.id_col].tolist()
        items = todo[input_col].tolist()

        async def _on_flush(local_indices: list[int], local_outputs: list[Any]):
            await self.store.flush(
                FlushBatch(
                    ids=[ids[i] for i in local_indices],
                    rows=[local_outputs[i] for i in local_indices],
                ),
                output_cols=output_cols,
            )

        if not hasattr(job, "transform_streaming"):
            raise TypeError(
                "Job must implement `transform_streaming(items, on_flush=..., checkpoint_every=...)`"
            )

        await job.transform_streaming(
            items,
            on_flush=_on_flush,
            checkpoint_every=self.cfg.checkpoint_every,
        )
        return self.store.finalize()


async def run_sharded(
    job_factory: Callable[[], Any],
    df: pd.DataFrame,
    *,
    input_col: str,
    output_cols: Optional[list[str]],
    store_uri: str,  # e.g. file:///..., s3://...
    nshards: int = 1,
    cfg: RunnerConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or RunnerConfig()
    if nshards <= 1:
        store = ParquetShardStore(store_uri)
        return await JobRunner(store, cfg).run(
            job_factory(), df, input_col=input_col, output_cols=output_cols
        )

    # split by index and process concurrently
    indices = np.array_split(df.index, nshards)
    import asyncio

    async def _one_shard(shard_id: int, idx):
        sub = df.loc[idx].copy()
        su = store_uri.replace(".parquet", f"_shard{shard_id}.parquet")
        store = ParquetShardStore(su)
        runner = JobRunner(store, cfg)
        return await runner.run(
            job_factory(), sub, input_col=input_col, output_cols=output_cols
        )

    parts = await asyncio.gather(*[_one_shard(i, idx) for i, idx in enumerate(indices)])
    return pd.concat(parts).sort_index()
