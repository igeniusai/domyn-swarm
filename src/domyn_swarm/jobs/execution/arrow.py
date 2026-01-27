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

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import logging
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from domyn_swarm.checkpoint.arrow_store import ArrowShardStore, InMemoryArrowStore
from domyn_swarm.checkpoint.store import FlushBatch
from domyn_swarm.jobs.api.base import OutputJoinMode, SwarmJob
from domyn_swarm.jobs.api.runner import normalize_batch_outputs
from domyn_swarm.jobs.io.checkpointing import (
    _shard_store_uri,
    _validate_checkpoint_store,
    _validate_sharded_execution,
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
        if mode == OutputJoinMode.REPLACE:
            return out_table

        # Build output columns aligned to the original order (APPEND/IO_ONLY).
        out_dict = out_table.to_pydict()
        out_ids = out_dict.get(self.cfg.id_col, [])
        output_columns = [c for c in out_table.column_names if c != self.cfg.id_col]
        out_rows = {
            out_ids[i]: {c: out_dict[c][i] for c in output_columns} for i in range(len(out_ids))
        }

        base_ids = table[self.cfg.id_col].to_pylist()
        if mode == OutputJoinMode.IO_ONLY:
            base_columns = [input_col]
            extra = [
                c
                for c in table.column_names
                if c not in (self.cfg.id_col, input_col) and c in output_columns
            ]
            base_columns.extend(extra)
        else:
            base_columns = [c for c in table.column_names if c != self.cfg.id_col]
        collisions = set(base_columns) & set(output_columns)

        def _base_name(col: str) -> str:
            return f"{col}_x" if col in collisions else col

        def _out_name(col: str) -> str:
            return f"{col}_y" if col in collisions else col

        columns: dict[str, list[Any]] = {self.cfg.id_col: base_ids}
        for c in base_columns:
            columns[_base_name(c)] = table[c].to_pylist()

        for c in output_columns:
            columns[_out_name(c)] = [out_rows.get(rid, {}).get(c) for rid in base_ids]

        return pa.Table.from_pydict(columns)


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
        store_uri: Base checkpoint store URI.
        checkpoint_every: Flush interval in items.
        checkpointing: Whether checkpointing is enabled.

    Returns:
        Arrow table containing job outputs (and inputs depending on output mode).
    """
    if isinstance(data, pa.Table):
        table = data
    elif isinstance(data, pd.DataFrame):
        df = data
        if id_col not in df.columns:
            df = df.copy(deep=False)
            df[id_col] = df.index
        table = backend.to_arrow(df)
    else:
        table = backend.to_arrow(data)
    if require_id and id_col not in table.column_names:
        raise ValueError(f"Input table missing required id column '{id_col}'.")
    _validate_checkpoint_store(checkpointing, store_uri)
    if nshards <= 1:
        return await run_arrow_job(
            job_factory,
            table,
            input_col=input_col,
            output_cols=output_cols,
            store_uri=store_uri,
            checkpoint_every=checkpoint_every,
            checkpointing=checkpointing,
            id_col=id_col,
        )
    _validate_sharded_execution(checkpointing)

    if id_col not in table.column_names:
        table = _ensure_arrow_id(table, id_col)
    indices = np.array_split(np.arange(table.num_rows), nshards)

    async def _one(i, idx):
        assert store_uri is not None
        sub = table.take(pa.array(idx, type=pa.int64()))
        su = _shard_store_uri(store_uri, i)
        return await run_arrow_job(
            job_factory,
            sub,
            input_col=input_col,
            output_cols=output_cols,
            store_uri=su,
            checkpoint_every=checkpoint_every,
            checkpointing=checkpointing,
            id_col=id_col,
        )

    parts = await asyncio.gather(*[_one(i, idx) for i, idx in enumerate(indices)])
    return pa.concat_tables(parts)
