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

import pyarrow as pa

from domyn_swarm.checkpoint.arrow_store import ArrowShardStore, InMemoryArrowStore
from domyn_swarm.checkpoint.store import FlushBatch
from domyn_swarm.jobs import OutputJoinMode, SwarmJob
from domyn_swarm.jobs.runner import _normalize_batch_outputs

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

        table = table
        if self.cfg.id_col not in table.column_names:
            # Inject id as positional index
            ids = pa.array(range(len(table)))
            table = table.append_column(self.cfg.id_col, ids)

        todo = self.store.prepare(table, self.cfg.id_col)
        ids_col = todo[self.cfg.id_col]
        ids = ids_col.to_pylist()
        items = todo[input_col].to_pylist()

        async def _on_flush(local_indices: list[int], local_outputs: list[Any]):
            batch_ids = [ids[i] for i in local_indices]
            rows, cols = _normalize_batch_outputs(local_outputs, output_cols)
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
        columns: dict[str, list[Any]] = {}

        # Start from the original columns depending on the mode.
        if mode == OutputJoinMode.IO_ONLY:
            columns[self.cfg.id_col] = base_ids
            columns[input_col] = table[input_col].to_pylist()
        else:  # APPEND
            for c in table.column_names:
                columns[c] = table[c].to_pylist()

        for c in output_columns:
            columns[c] = [out_rows.get(rid, {}).get(c) for rid in base_ids]

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
