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

"""Compatibility shim for deprecated imports."""

from __future__ import annotations

import asyncio
from typing import cast
import warnings

import numpy as np
import pandas as pd

from domyn_swarm.checkpoint.store import ParquetShardStore
from domyn_swarm.jobs.api.base import OutputJoinMode
from domyn_swarm.jobs.api.runner import JobRunner, RunnerConfig, normalize_batch_outputs
from domyn_swarm.jobs.io.sharding import shard_indices_by_id

warnings.warn(
    "domyn_swarm.jobs.runner is deprecated; use domyn_swarm.jobs.api.runner",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "JobRunner",
    "OutputJoinMode",
    "ParquetShardStore",
    "RunnerConfig",
    "normalize_batch_outputs",
    "run_sharded",
]


async def run_sharded(
    job_factory,
    df: pd.DataFrame,
    *,
    input_col: str,
    output_cols: list[str] | None,
    store_uri: str,
    nshards: int = 1,
    cfg: RunnerConfig | None = None,
    shard_mode: str = "id",
):
    """Run a job in sharded mode using the legacy runner facade.

    Args:
        job_factory: Callable producing a job instance.
        df: Input DataFrame.
        input_col: Column name for inputs.
        output_cols: Output column names (None for dict outputs).
        store_uri: Base checkpoint store URI.
        nshards: Number of shards to split the input into.
        cfg: Optional RunnerConfig override.
        shard_mode: Sharding strategy ("id" for stable id hashing, "index" for legacy order).

    Returns:
        DataFrame with merged outputs.
    """
    cfg = cfg or RunnerConfig()
    if nshards <= 1:
        store = ParquetShardStore(store_uri)
        return await JobRunner(store, cfg).run(
            job_factory(), df, input_col=input_col, output_cols=output_cols
        )

    if shard_mode not in {"id", "index"}:
        raise ValueError(f"Unsupported shard_mode: {shard_mode}")
    if shard_mode == "index":
        indices = np.array_split(df.index, nshards)

        def _slice(idx):
            return df.loc[idx].copy(deep=False)

    else:
        id_col = cfg.id_col
        if id_col not in df.columns:
            df = df.copy(deep=False)
            df[id_col] = df.index
        ids = cast(pd.Series, df[id_col])
        indices = shard_indices_by_id(ids, nshards)

        def _slice(idx):
            return df.iloc[idx].copy(deep=False)

    async def _one_shard(shard_id: int, idx):
        sub = _slice(idx)
        su = store_uri.replace(".parquet", f"_shard{shard_id}.parquet")
        store = ParquetShardStore(su)
        runner = JobRunner(store, cfg)
        return await runner.run(job_factory(), sub, input_col=input_col, output_cols=output_cols)

    parts = await asyncio.gather(*[_one_shard(i, idx) for i, idx in enumerate(indices)])
    return pd.concat(parts).sort_index()
