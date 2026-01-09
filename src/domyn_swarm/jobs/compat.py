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

import asyncio
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from domyn_swarm.checkpoint.store import ParquetShardStore
from domyn_swarm.jobs.base import SwarmJob
from domyn_swarm.jobs.runner import JobRunner, RunnerConfig


def _has_new_api(job: SwarmJob) -> bool:
    return resolve_job_api(job) == "new"


def _is_overridden(obj: object, method: str, base: type) -> bool:
    """
    True iff `method` is implemented by `type(obj)` (or an intermediate base),
    not by `base` itself.
    """
    cls = type(obj)
    for c in cls.mro():
        if method in c.__dict__:
            return c is not base  # first definition found is not the base → overridden
    return False  # method not found at all (shouldn't happen here)


def resolve_job_api(job: SwarmJob) -> str:
    if getattr(type(job), "api_version", 1) >= 2:
        return "new"
    # fallback to override-based inference for older subclasses
    if _is_overridden(job, "transform_items", SwarmJob) or _is_overridden(
        job, "transform_streaming", SwarmJob
    ):
        return "new"
    if _is_overridden(job, "transform", SwarmJob) or _is_overridden(job, "run", SwarmJob):
        return "old"
    return "old"


async def run_job_unified(
    job_factory: Callable[[], Any],
    df: pd.DataFrame,
    *,
    input_col: str,
    output_cols: list[str] | None,
    nshards: int = 1,
    store_uri: str | None = None,
    checkpoint_every: int = 16,
    tag: str | None = None,
    checkpoint_dir: str | None = None,
) -> pd.DataFrame:
    job_probe = job_factory()

    if not _has_new_api(job_probe):
        raise TypeError(
            "Job must implement the streaming API (transform_items/transform_streaming)."
        )

    if store_uri is None:
        raise ValueError("store_uri is required for streaming jobs.")
    cfg = RunnerConfig(id_col="_row_id", checkpoint_every=checkpoint_every)
    if nshards <= 1:
        store = ParquetShardStore(store_uri)
        return await JobRunner(store, cfg).run(
            job_probe,
            df,
            input_col=input_col,
            output_cols=output_cols or job_probe.default_output_cols,
            output_mode=job_probe.output_mode,
        )
    indices = np.array_split(df.index, nshards)

    async def _one(i, idx):
        sub = df.loc[idx].copy()
        su = store_uri.replace(".parquet", f"_shard{i}.parquet")
        store = ParquetShardStore(su)
        return await JobRunner(store, cfg).run(
            job_factory(),
            sub,
            input_col=input_col,
            output_cols=output_cols or job_probe.default_output_cols,
            output_mode=job_probe.output_mode,
        )

    parts = await asyncio.gather(*[_one(i, idx) for i, idx in enumerate(indices)])
    return pd.concat(parts).sort_index()
