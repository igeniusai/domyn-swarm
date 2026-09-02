# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import logging
from typing import Any

import pyarrow as pa

from domyn_swarm.checkpoint.arrow_store import (
    ArrowShardStore,
    InMemoryArrowStore,
    _concat_tables_with_variable_width_fallback,
    _take_with_offset_overflow_fallback,
)
from domyn_swarm.checkpoint.store import FlushBatch
from domyn_swarm.jobs.api.base import OutputJoinMode, SwarmJob
from domyn_swarm.jobs.api.runner import normalize_batch_outputs
from domyn_swarm.jobs.execution.frame_ops import ArrowFrameOps, ShardSpec
from domyn_swarm.jobs.execution.pipeline import run_sharded_pipeline
from domyn_swarm.jobs.io.checkpointing import (
    _shard_store_uri,
)

logger = logging.getLogger(__name__)


@dataclass
class ArrowRunnerConfig:
    """Configuration for ArrowJobRunner."""

    id_col: str = "_row_id"
    checkpoint_every: int = 16


class ArrowJobRunner:
    """
    Arrow-native runner that mirrors JobRunner semantics (APPEND/REPLACE/IO_ONLY) but
    operates on pyarrow.Table internally.
    """

    def __init__(
        self, store: ArrowShardStore | InMemoryArrowStore, cfg: ArrowRunnerConfig | None = None
    ):
        self.store = store
        self.cfg = cfg or ArrowRunnerConfig()

    async def run(
        self,
        job: SwarmJob,
        table: pa.Table,
        *,
        input_col: str,
        output_cols: list[str] | None,
        output_mode: OutputJoinMode | None = None,
    ) -> pa.Table:
        """Run a SwarmJob against an Arrow table.

        Args:
            job: SwarmJob instance to execute.
            table: Input Arrow table.
            input_col: Column name to read inputs from.
            output_cols: Output column names (None for dict outputs).
            output_mode: Output join mode (APPEND, REPLACE, IO_ONLY).

        Returns:
            Arrow table with job outputs (and inputs depending on mode).
        """
        if input_col not in table.column_names:
            raise ValueError(f"input_col '{input_col}' not found in table")

        mode = output_mode or getattr(job, "output_mode", OutputJoinMode.APPEND)
        if isinstance(mode, str):
            mode = OutputJoinMode(mode)

        table = _ensure_arrow_id(table, self.cfg.id_col)

        todo = self.store.prepare(table, self.cfg.id_col)
        ids_col = todo[self.cfg.id_col]
        ids = ids_col.to_pylist()
        items = todo[input_col].to_pylist()

        async def _on_flush(local_indices: list[int], local_outputs: list[Any]):
            batch_ids = [ids[i] for i in local_indices]
            rows, cols = normalize_batch_outputs(local_outputs, output_cols)
            await self.store.flush(
                FlushBatch(ids=batch_ids, rows=rows),
                output_cols=cols,
            )

        if not hasattr(job, "transform_streaming"):
            raise TypeError(
                "Job must implement "
                "`transform_streaming(items, on_flush=..., checkpoint_every=...)`"
            )

        await job.transform_streaming(
            items,
            on_flush=_on_flush,
            checkpoint_every=self.cfg.checkpoint_every,
        )

        out_table = await asyncio.to_thread(self.store.finalize)
        return _join_arrow_outputs(
            table=table,
            out_table=out_table,
            input_col=input_col,
            output_mode=mode,
            id_col=self.cfg.id_col,
        )


async def run_arrow_job(
    job_factory: Callable[[], Any],
    table: pa.Table,
    *,
    input_col: str,
    output_cols: list[str] | None,
    store_uri: str | None,
    checkpoint_every: int,
    checkpointing: bool,
    id_col: str,
) -> pa.Table:
    """Run a job using the Arrow runner and an Arrow-aware checkpoint store.

    Args:
        job_factory: Callable that returns a SwarmJob instance.
        table: Input Arrow table.
        input_col: Input column name.
        output_cols: Output column names (None for dict outputs).
        store_uri: Base checkpoint store URI (required if checkpointing).
        checkpoint_every: Flush interval in items.
        checkpointing: Whether to read/write checkpoint state.
        id_col: Column name used for stable row ids.

    Returns:
        Arrow table containing job outputs.
    """
    store = ArrowShardStore(store_uri) if checkpointing and store_uri else InMemoryArrowStore()
    cfg = ArrowRunnerConfig(id_col=id_col, checkpoint_every=checkpoint_every)
    runner = ArrowJobRunner(store, cfg)
    job = job_factory()
    return await runner.run(
        job,
        table,
        input_col=input_col,
        output_cols=output_cols,
        output_mode=getattr(job, "output_mode", OutputJoinMode.APPEND),
    )


def _join_arrow_outputs(
    *,
    table: pa.Table,
    out_table: pa.Table,
    input_col: str,
    output_mode: OutputJoinMode | str | None,
    id_col: str,
) -> pa.Table:
    """Join Arrow outputs against the base table according to output mode.

    Args:
        table: Full input table with the id column.
        out_table: Output table containing the id column.
        input_col: Input column name.
        output_mode: Output join mode (APPEND, REPLACE, IO_ONLY).
        id_col: Column name containing stable ids.

    Returns:
        Arrow table with outputs joined to the base table.
    """
    mode = output_mode or OutputJoinMode.APPEND
    if isinstance(mode, str):
        mode = OutputJoinMode(mode)
    if mode == OutputJoinMode.REPLACE:
        return out_table

    out_dict = out_table.to_pydict()
    out_ids = out_dict.get(id_col, [])
    output_columns = [c for c in out_table.column_names if c != id_col]
    out_rows = {
        out_ids[i]: {c: out_dict[c][i] for c in output_columns} for i in range(len(out_ids))
    }

    base_ids = table[id_col].to_pylist()
    if mode == OutputJoinMode.IO_ONLY:
        base_columns = [input_col]
        extra = [
            c for c in table.column_names if c not in (id_col, input_col) and c in output_columns
        ]
        base_columns.extend(extra)
    else:
        base_columns = [c for c in table.column_names if c != id_col]

    collisions = set(base_columns) & set(output_columns)

    def _base_name(col: str) -> str:
        return f"{col}_x" if col in collisions else col

    def _out_name(col: str) -> str:
        return f"{col}_y" if col in collisions else col

    columns: dict[str, list[Any]] = {id_col: base_ids}
    for c in base_columns:
        columns[_base_name(c)] = table[c].to_pylist()

    for c in output_columns:
        columns[_out_name(c)] = [out_rows.get(rid, {}).get(c) for rid in base_ids]

    return pa.Table.from_pydict(columns)


def _ensure_arrow_id(table: pa.Table, id_col: str) -> pa.Table:
    """Ensure an Arrow table contains the id column.

    Args:
        table: Input Arrow table.
        id_col: Column name for row ids.

    Returns:
        Arrow table with the id column present.
    """
    if id_col in table.column_names:
        return table
    for candidate in ("__index_level_0__", "index", "level_0"):
        if candidate in table.column_names:
            return table.rename_columns(
                [id_col if c == candidate else c for c in table.column_names]
            )
    ids = pa.array(range(len(table)))
    return table.append_column(id_col, ids)


def _merge_shard_outputs(*, store_uri: str, nshards: int, id_col: str) -> pa.Table:
    """Merge all shard checkpoint outputs into a single Arrow table.

    Args:
        store_uri: Base checkpoint store URI.
        nshards: Number of shards.
        id_col: Column name containing stable ids.

    Returns:
        Arrow table with de-duplicated outputs across shards.
    """
    tables: list[pa.Table] = []
    for shard_id in range(nshards):
        shard_store = ArrowShardStore(_shard_store_uri(store_uri, shard_id), id_col=id_col)
        shard_table = shard_store.finalize()
        if shard_table.num_rows:
            tables.append(shard_table)
    if not tables:
        return pa.Table.from_pydict({id_col: []})

    merged = _concat_tables_with_variable_width_fallback(tables)
    ids = merged.column(id_col).to_pylist()
    last_indices = {value: idx for idx, value in enumerate(ids)}
    keep_indices = sorted(last_indices.values())
    return _take_with_offset_overflow_fallback(merged, keep_indices)


def _finalize_arrow_global_resume(
    *,
    table_full: pa.Table,
    store_uri: str,
    nshards: int,
    id_col: str,
    input_col: str,
    output_mode: OutputJoinMode | str | None,
) -> pa.Table:
    """Finalize sharded Arrow outputs for global resume.

    Args:
        table_full: Full input Arrow table (pre-filter).
        store_uri: Base checkpoint store URI.
        nshards: Number of shards.
        id_col: Column name containing stable ids.
        input_col: Input column name.
        output_mode: Output join mode (APPEND, REPLACE, IO_ONLY).

    Returns:
        Arrow table containing joined outputs.
    """
    merged = _merge_shard_outputs(store_uri=store_uri, nshards=nshards, id_col=id_col)
    return _join_arrow_outputs(
        table=table_full,
        out_table=merged,
        input_col=input_col,
        output_mode=output_mode,
        id_col=id_col,
    )


async def _run_arrow(
    *,
    job_factory: Callable[[], Any],
    backend: Any,
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
) -> pa.Table:
    """Run the Arrow runner path for non-ray backends.

    Args:
        job_factory: Callable producing a SwarmJob instance.
        backend: Data backend used for Arrow conversion.
        data: Backend-native data or Arrow table.
        input_col: Input column name in the dataset.
        output_cols: Output columns to produce (None for dict outputs).
        id_col: Column name used for stable row ids.
        require_id: Whether id_col must already exist in the input.
        nshards: Number of shards to split the input into.
        shard_mode: Sharding strategy ("id" for stable id hashing, "index" for legacy order).
        global_resume: Whether to resume using global done ids across shards.
        store_uri: Base checkpoint store URI.
        checkpoint_every: Flush interval in items.
        checkpointing: Whether checkpointing is enabled.

    Returns:
        Arrow table containing job outputs (and inputs depending on output mode).
    """
    ops = ArrowFrameOps(backend)
    spec = ShardSpec(
        input_col=input_col,
        output_cols=output_cols,
        id_col=id_col,
        checkpoint_every=checkpoint_every,
        checkpointing=checkpointing,
        output_mode=getattr(job_factory(), "output_mode", OutputJoinMode.APPEND),
    )

    def _finalize(*, frame_full, store_uri, nshards, spec) -> pa.Table:
        return _finalize_arrow_global_resume(
            table_full=frame_full,
            store_uri=store_uri,
            nshards=nshards,
            id_col=spec.id_col,
            input_col=spec.input_col,
            output_mode=spec.output_mode,
        )

    return await run_sharded_pipeline(
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
