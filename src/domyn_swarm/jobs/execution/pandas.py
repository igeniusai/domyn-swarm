# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from domyn_swarm.checkpoint.store import ParquetShardStore
from domyn_swarm.data.backends.base import DataBackend
from domyn_swarm.jobs.api.base import OutputJoinMode, SwarmJob
from domyn_swarm.jobs.api.runner import RunnerConfig
from domyn_swarm.jobs.execution.frame_ops import PandasFrameOps, ShardSpec
from domyn_swarm.jobs.execution.pipeline import run_sharded_pipeline
from domyn_swarm.jobs.io.checkpointing import (
    _shard_filename,
    _shard_store_uri,
    _validate_sharded_execution,
    load_global_done_ids,
)


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
    ops = PandasFrameOps(backend)
    resolved_output_cols = output_cols or job_probe.default_output_cols
    spec = ShardSpec(
        input_col=input_col,
        output_cols=resolved_output_cols,
        id_col=id_col,
        checkpoint_every=checkpoint_every,
        checkpointing=checkpointing,
        output_mode=job_probe.output_mode,
    )
    is_dir_output = output_path is not None and (output_path.is_dir() or output_path.suffix == "")

    if backend.name == "pandas" and is_dir_output and nshards > 1:
        assert output_path is not None
        return await _run_pandas_to_directory(
            ops=ops,
            job_factory=job_factory,
            data=data,
            spec=spec,
            require_id=require_id,
            nshards=nshards,
            shard_mode=shard_mode,
            global_resume=global_resume,
            store_uri=store_uri,
            checkpointing=checkpointing,
            output_path=output_path,
        )

    def _finalize(
        *, frame_full: pd.DataFrame, store_uri: str | None, nshards: int, spec: ShardSpec
    ) -> pd.DataFrame:
        return _finalize_global_resume(
            df_full=frame_full,
            store_uri=store_uri,
            nshards=nshards,
            cfg=RunnerConfig(id_col=spec.id_col, checkpoint_every=spec.checkpoint_every),
            input_col=spec.input_col,
            # `spec.output_cols` is typed `list[str] | None`, but the closure over the
            # already-resolved local avoids re-deriving (or asserting) that it isn't
            # None here: it's the same value `spec` was built from.
            resolved_output_cols=resolved_output_cols,
            output_mode=spec.output_mode,
        )

    out = await run_sharded_pipeline(
        ops=ops,
        job_factory=job_factory,
        data=data,
        spec=spec,
        require_id=require_id,
        nshards=nshards,
        shard_mode=shard_mode,
        global_resume=global_resume,
        store_uri=store_uri,
        finalize=_finalize,
    )
    return out if backend.name == "pandas" else backend.from_pandas(out)


async def _run_pandas_to_directory(
    *,
    ops: PandasFrameOps,
    job_factory: Callable[[], Any],
    data: Any,
    spec: ShardSpec,
    require_id: bool,
    nshards: int,
    shard_mode: str,
    global_resume: bool,
    store_uri: str | None,
    checkpointing: bool,
    output_path: Path,
) -> None:
    """Write one parquet file per shard into a directory.

    Retained from the pre-consolidation pandas engine, where directory output
    was triggered implicitly by `output_path` being a directory.

    The validation order mirrors `_run_pandas`: the id column is checked
    right after coercion (before `ensure_id` would paper over a missing one),
    then sharded-execution prerequisites, then the frame is prepared for
    sharding (id backfill, global-resume filtering, shard partitioning).

    Args:
        ops: Pandas frame adapter.
        job_factory: Callable producing a fresh job per shard.
        data: Backend-native input data.
        spec: Settings shared by every shard.
        require_id: Whether the id column must already exist.
        nshards: Number of shards.
        shard_mode: Sharding strategy.
        global_resume: Whether to skip globally-done ids.
        store_uri: Base checkpoint store URI.
        checkpointing: Whether checkpointing is enabled.
        output_path: Directory to write shard parquet files into.
    """
    frame = ops.coerce(data, spec.id_col)
    if require_id and spec.id_col not in ops.column_names(frame):
        raise ValueError(f"Input is missing required id column {spec.id_col!r}.")

    _validate_sharded_execution(checkpointing)
    if shard_mode not in {"id", "index"}:
        raise ValueError(f"Unsupported shard_mode: {shard_mode}")
    if store_uri is None:
        raise ValueError("store_uri is required when running sharded jobs.")

    frame = ops.ensure_id(frame, spec.id_col)

    if global_resume and checkpointing:
        done = load_global_done_ids(
            store_uri=store_uri,
            id_col=spec.id_col,
            nshards=nshards,
            store_factory=ops.store_factory,
            empty_data_factory=lambda: ops.empty_with_id(spec.id_col),
        )
        frame = ops.filter_out_ids(frame, spec.id_col, done)

    indices = ops.shard_indices(frame, spec.id_col, shard_mode, nshards)
    take = ops.take if shard_mode == "index" else ops.take_positional
    output_path.mkdir(parents=True, exist_ok=True)

    async def _run_shard(i: int, idx: np.ndarray) -> pd.DataFrame:
        assert store_uri is not None
        return await ops.run_shard(
            job_factory, take(frame, idx), _shard_store_uri(store_uri, i), spec
        )

    await _write_sharded_outputs(
        indices=indices, nshards=nshards, output_path=output_path, run_shard=_run_shard
    )
    return None


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
) -> pd.DataFrame:
    """Rebuild outputs from all shards and join against the full input.

    Args:
        df_full: Original input DataFrame (before filtering).
        store_uri: Checkpoint store URI.
        nshards: Number of shards.
        cfg: Runner configuration.
        input_col: Input column name.
        resolved_output_cols: Output columns.
        output_mode: Output join mode.

    Returns:
        Final output DataFrame, in pandas form. The caller (`_run_pandas`)
        is responsible for converting to the target backend exactly once.
    """
    if store_uri is None:
        raise ValueError("store_uri is required when global_resume is enabled.")
    merged_parts: list[pd.DataFrame] = []
    for shard_id in range(nshards):
        shard_uri = _shard_store_uri(store_uri, shard_id)
        shard_store = ParquetShardStore(shard_uri, id_col=cfg.id_col)
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
        return merged
    if output_mode == OutputJoinMode.IO_ONLY:
        merged = base.merge(out_df, on=cfg.id_col, how="left")
        if resolved_output_cols:
            keep = [cfg.id_col, input_col, *resolved_output_cols]
        else:
            output_columns = [c for c in merged.columns if c not in (cfg.id_col, input_col)]
            keep = [cfg.id_col, input_col, *output_columns]
        merged = merged.loc[:, keep]
        return merged
    if resolved_output_cols:
        keep = [cfg.id_col, *resolved_output_cols]
    else:
        output_columns = [c for c in out_df.columns if c != cfg.id_col]
        keep = [cfg.id_col, *output_columns]
    merged = out_df.loc[:, keep]
    return merged
