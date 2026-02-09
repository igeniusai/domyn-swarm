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
from typing import Any, cast

import numpy as np
import pandas as pd

from domyn_swarm.checkpoint.store import ParquetShardStore
from domyn_swarm.data.backends.base import DataBackend
from domyn_swarm.jobs.api.base import OutputJoinMode, SwarmJob
from domyn_swarm.jobs.api.runner import JobRunner, RunnerConfig
from domyn_swarm.jobs.io.checkpointing import (
    _build_checkpoint_store,
    _shard_filename,
    _shard_store_uri,
    _validate_sharded_execution,
    load_global_done_ids,
)
from domyn_swarm.jobs.io.columns import _validate_required_id
from domyn_swarm.jobs.io.sharding import shard_indices_by_id


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
    shard_mode: str,
    global_resume: bool,
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
        shard_mode: Sharding strategy ("id" for stable id hashing, "index" for legacy order).
        global_resume: Whether to resume using global done ids across shards.
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
    if shard_mode not in {"id", "index"}:
        raise ValueError(f"Unsupported shard_mode: {shard_mode}")

    df_full, df = _prepare_sharded_inputs(
        df=df,
        id_col=id_col,
        nshards=nshards,
        global_resume=global_resume,
        checkpointing=checkpointing,
        store_uri=store_uri,
    )
    indices, _slice = _build_shard_slices(
        df=df, id_col=id_col, shard_mode=shard_mode, nshards=nshards
    )

    async def _run_shard(i: int, idx):
        sub = _slice(idx)
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
        await _write_sharded_outputs(
            indices=indices,
            nshards=nshards,
            output_path=output_path,
            run_shard=_run_shard,
        )
        return None

    parts = await asyncio.gather(*[_run_shard(i, idx) for i, idx in enumerate(indices)])
    out = pd.concat(parts).sort_index()
    if not global_resume:
        return out if backend.name == "pandas" else backend.from_pandas(out)
    return _finalize_global_resume(
        df_full=df_full,
        store_uri=store_uri,
        nshards=nshards,
        cfg=cfg,
        input_col=input_col,
        resolved_output_cols=resolved_output_cols,
        output_mode=job_probe.output_mode,
        backend=backend,
    )


def _prepare_sharded_inputs(
    *,
    df: pd.DataFrame,
    id_col: str,
    nshards: int,
    global_resume: bool,
    checkpointing: bool,
    store_uri: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare inputs for sharded execution with optional global resume.

    Args:
        df: Input DataFrame.
        id_col: Column name used for stable row ids.
        nshards: Number of shards.
        global_resume: Whether to filter by global done ids.
        checkpointing: Whether checkpointing is enabled.
        store_uri: Checkpoint store URI.

    Returns:
        Tuple of (full_df, filtered_df) where filtered_df excludes done ids if requested.
    """
    if id_col not in df.columns:
        df = df.copy(deep=False)
        df[id_col] = df.index
    df_full = df
    if global_resume and checkpointing:
        if store_uri is None:
            raise ValueError("store_uri is required when global_resume is enabled.")
        done_ids = load_global_done_ids(
            store_uri=store_uri,
            id_col=id_col,
            nshards=nshards,
            store_factory=ParquetShardStore,
            empty_data_factory=lambda: pd.DataFrame({id_col: []}),
        )
        if done_ids:
            df = df.loc[~df[id_col].isin(list(done_ids))]
    return df_full, df


def _build_shard_slices(
    *,
    df: pd.DataFrame,
    id_col: str,
    shard_mode: str,
    nshards: int,
) -> tuple[list[np.ndarray], Callable[[np.ndarray], pd.DataFrame]]:
    """Build shard indices and a slice function for the DataFrame.

    Args:
        df: Input DataFrame.
        id_col: Column name used for stable row ids.
        shard_mode: Sharding strategy ("id" or "index").
        nshards: Number of shards.

    Returns:
        List of index arrays and a callable that slices the DataFrame by index array.
    """
    if shard_mode == "index":
        indices = np.array_split(df.index, nshards)

        def _slice(idx: np.ndarray) -> pd.DataFrame:
            return df.loc[idx].copy(deep=False)

        return indices, _slice

    ids = cast(pd.Series, df[id_col])
    indices = shard_indices_by_id(ids, nshards)

    def _slice(idx: np.ndarray) -> pd.DataFrame:
        return df.iloc[idx].copy(deep=False)

    return indices, _slice


async def _write_sharded_outputs(
    *,
    indices: list[np.ndarray],
    nshards: int,
    output_path: Path,
    run_shard: Callable[[int, np.ndarray], Any],
) -> None:
    """Write one parquet file per shard to an output directory.

    Args:
        indices: Per-shard index arrays.
        nshards: Number of shards.
        output_path: Directory to write parquet shards into.
        run_shard: Async callable that runs a shard and returns a DataFrame.
    """

    async def _write_shard(i: int, idx: np.ndarray) -> None:
        part = await run_shard(i, idx)
        part.to_parquet(output_path / _shard_filename(i, nshards), index=False)

    await asyncio.gather(*[_write_shard(i, idx) for i, idx in enumerate(indices)])


def _finalize_global_resume(
    *,
    df_full: pd.DataFrame,
    store_uri: str | None,
    nshards: int,
    cfg: RunnerConfig,
    input_col: str,
    resolved_output_cols: list[str],
    output_mode: OutputJoinMode,
    backend: DataBackend,
) -> Any:
    """Rebuild outputs from all shards and join against the full input.

    Args:
        df_full: Original input DataFrame (before filtering).
        store_uri: Checkpoint store URI.
        nshards: Number of shards.
        cfg: Runner configuration.
        input_col: Input column name.
        resolved_output_cols: Output columns.
        output_mode: Output join mode.
        backend: Data backend used for conversions.

    Returns:
        Final output DataFrame (backend-native).
    """
    if store_uri is None:
        raise ValueError("store_uri is required when global_resume is enabled.")
    merged_parts: list[pd.DataFrame] = []
    for shard_id in range(nshards):
        shard_uri = _shard_store_uri(store_uri, shard_id)
        shard_store = ParquetShardStore(shard_uri)
        merged_parts.append(shard_store.finalize())
    if merged_parts:
        out_df = pd.concat(merged_parts)
        out_df = out_df[~out_df.index.duplicated(keep="last")]
    else:
        out_df = pd.DataFrame().set_index(cfg.id_col)

    if out_df.index.name == cfg.id_col:
        out_df = (
            out_df.reset_index(drop=True) if cfg.id_col in out_df.columns else out_df.reset_index()
        )

    base = df_full
    if cfg.id_col not in base.columns:
        base = base.copy(deep=False)
        base[cfg.id_col] = base.index
    if output_mode == OutputJoinMode.APPEND:
        merged = base.merge(out_df, on=cfg.id_col, how="left")
        return merged if backend.name == "pandas" else backend.from_pandas(merged)
    if output_mode == OutputJoinMode.IO_ONLY:
        merged = base.merge(out_df, on=cfg.id_col, how="left")
        if resolved_output_cols:
            keep = [cfg.id_col, input_col, *resolved_output_cols]
        else:
            output_columns = [c for c in merged.columns if c not in (cfg.id_col, input_col)]
            keep = [cfg.id_col, input_col, *output_columns]
        merged = merged.loc[:, keep]
        return merged if backend.name == "pandas" else backend.from_pandas(merged)
    if resolved_output_cols:
        keep = [cfg.id_col, *resolved_output_cols]
    else:
        output_columns = [c for c in out_df.columns if c != cfg.id_col]
        keep = [cfg.id_col, *output_columns]
    merged = out_df.loc[:, keep]
    return merged if backend.name == "pandas" else backend.from_pandas(merged)
