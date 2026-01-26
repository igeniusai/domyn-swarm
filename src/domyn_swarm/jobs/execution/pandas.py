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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from domyn_swarm.checkpoint.store import ParquetShardStore
from domyn_swarm.data.backends.base import DataBackend
from domyn_swarm.jobs.api.base import SwarmJob
from domyn_swarm.jobs.api.runner import JobRunner, RunnerConfig
from domyn_swarm.jobs.io.checkpointing import (
    _build_checkpoint_store,
    _shard_filename,
    _shard_store_uri,
    _validate_sharded_execution,
)
from domyn_swarm.jobs.io.columns import _validate_required_id


async def _run_pandas(
    *,
    job_factory: Callable[[], Any],
    job_probe: SwarmJob,
    backend: DataBackend,
    data: Any,
    input_col: str,
    output_cols: list[str] | None,
    id_col: str,
    require_id: bool,
    nshards: int,
    store_uri: str | None,
    checkpoint_every: int,
    checkpointing: bool,
    output_path: Path | None,
) -> Any:
    """Run the pandas-backed execution path for non-ray backends.

    Args:
        job_factory: Callable producing a SwarmJob instance.
        job_probe: Probe job instance for defaults.
        backend: Data backend used for conversion.
        data: Backend-native data or DataFrame.
        input_col: Input column name.
        output_cols: Output column names (None for dict outputs).
        id_col: Column name used for stable row ids.
        require_id: Whether id_col must already exist in the input.
        nshards: Number of shards to split the input into.
        store_uri: Base checkpoint store URI.
        checkpoint_every: Flush interval in items.
        checkpointing: Whether checkpointing is enabled.
        output_path: Optional output path used for direct shard writes.

    Returns:
        Job results in backend-native output form.
    """
    df = data if isinstance(data, pd.DataFrame) else backend.to_pandas(data)
    if require_id:
        _validate_required_id(df, id_col)

    cfg = RunnerConfig(id_col=id_col, checkpoint_every=checkpoint_every)
    resolved_output_cols = output_cols or job_probe.default_output_cols
    is_dir_output = output_path is not None and (output_path.is_dir() or output_path.suffix == "")

    if nshards <= 1:
        store = _build_checkpoint_store(checkpointing=checkpointing, store_uri=store_uri)
        out = await JobRunner(store, cfg).run(
            job_probe,
            df,
            input_col=input_col,
            output_cols=resolved_output_cols,
            output_mode=job_probe.output_mode,
        )
        return out if backend.name == "pandas" else backend.from_pandas(out)

    _validate_sharded_execution(checkpointing)
    indices = np.array_split(df.index, nshards)

    async def _run_shard(i: int, idx):
        sub = df.loc[idx].copy(deep=False)
        assert store_uri is not None
        su = _shard_store_uri(store_uri, i)
        store = ParquetShardStore(su)
        return await JobRunner(store, cfg).run(
            job_factory(),
            sub,
            input_col=input_col,
            output_cols=resolved_output_cols,
            output_mode=job_probe.output_mode,
        )

    if backend.name == "pandas" and is_dir_output:
        assert output_path is not None
        output_path.mkdir(parents=True, exist_ok=True)

        async def _write_shard(i: int, idx) -> None:
            part = await _run_shard(i, idx)
            part.to_parquet(output_path / _shard_filename(i, nshards), index=False)

        await asyncio.gather(*[_write_shard(i, idx) for i, idx in enumerate(indices)])
        return None

    parts = await asyncio.gather(*[_run_shard(i, idx) for i, idx in enumerate(indices)])
    out = pd.concat(parts).sort_index()
    return out if backend.name == "pandas" else backend.from_pandas(out)
