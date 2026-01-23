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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pyarrow as pa

from domyn_swarm.checkpoint.arrow_store import ArrowShardStore, InMemoryArrowStore
from domyn_swarm.checkpoint.store import FlushBatch
from domyn_swarm.data.backends.base import DataBackend
from domyn_swarm.jobs.base import OutputJoinMode, SwarmJob
from domyn_swarm.jobs.runner import _normalize_batch_outputs

if TYPE_CHECKING:
    import polars as pl


@dataclass
class PolarsRunnerConfig:
    """Configuration for PolarsJobRunner."""

    id_col: str = "_row_id"
    checkpoint_every: int = 16
    batch_size: int | None = None


class PolarsJobRunner:
    """Polars-native runner that processes inputs via backend `iter_batches`.

    This runner avoids pandas conversions by:
    - keeping the input dataset as a polars DataFrame/LazyFrame
    - iterating polars DataFrame batches via `DataBackend.iter_batches`
    - checkpointing outputs via `ArrowShardStore` (parquet shards)

    Notes:
        For LazyFrame inputs (e.g. created via `polars.scan_parquet`), batching is driven by the
        backend's `iter_batches`, which may execute the query in streaming mode and yield
        materialized DataFrame batches. The final output join still requires a full scan/collect.
    """

    def __init__(
        self,
        store: ArrowShardStore | InMemoryArrowStore,
        backend: DataBackend,
        cfg: PolarsRunnerConfig | None = None,
    ):
        """Initialize the runner with a checkpoint store and a backend.

        Args:
            store: Checkpoint store used for resume and flushes.
            backend: Data backend providing `iter_batches`.
            cfg: Optional configuration for id column and batch sizing.
        """
        self.store = store
        self.backend = backend
        self.cfg = cfg or PolarsRunnerConfig()

    def _column_names(self, data: pl.DataFrame | pl.LazyFrame) -> list[str]:
        """Return column names without forcing an expensive schema resolution.

        Args:
            data: Polars DataFrame or LazyFrame.

        Returns:
            List of column names.
        """
        import polars as pl

        if isinstance(data, pl.LazyFrame):
            return data.collect_schema().names()
        return list(data.columns)

    def _ensure_id(self, data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        """Ensure `id_col` exists on the dataset, injecting it if needed.

        Args:
            data: Polars DataFrame or LazyFrame.

        Returns:
            Data with an `id_col` column.
        """
        if self.cfg.id_col in self._column_names(data):
            return data
        return data.with_row_index(self.cfg.id_col)

    def _scan_checkpoint_outputs(self) -> pl.LazyFrame | None:
        """Build a LazyFrame over checkpoint outputs if available.

        Returns:
            LazyFrame over checkpoint outputs, or None when no outputs exist.
        """
        import polars as pl

        store = self.store
        if not isinstance(store, ArrowShardStore):
            return None

        unstrip = getattr(store.fs, "unstrip_protocol", None)

        def _with_protocol(path: str) -> str:
            if callable(unstrip):
                return cast(str, unstrip(path))
            return path

        parts = sorted(store.fs.glob(store.dir_path + "*.parquet"))
        if parts:
            part_paths = [_with_protocol(p) for p in parts]
            return pl.scan_parquet(part_paths)
        if store.fs.exists(store.base_path):
            return pl.scan_parquet(_with_protocol(store.base_path))
        return None

    def _stream_output_to_dir(
        self,
        *,
        data: pl.DataFrame | pl.LazyFrame,
        input_col: str,
        mode: OutputJoinMode,
        output_path: Path,
    ) -> bool:
        """Stream output directly to a directory when possible.

        Args:
            data: Input data (DataFrame or LazyFrame).
            input_col: Input column name.
            mode: Output join mode.
            output_path: Directory to write parquet outputs into.

        Returns:
            True if output was streamed to the directory, False otherwise.
        """
        import polars as pl

        out_lf = self._scan_checkpoint_outputs()
        if out_lf is None:
            return False

        output_path.mkdir(parents=True, exist_ok=True)
        if mode == OutputJoinMode.REPLACE:
            out_lf.sink_parquet(output_path)
            return True

        base_lf = (
            data.select([self.cfg.id_col, input_col]) if mode == OutputJoinMode.IO_ONLY else data
        )
        if isinstance(base_lf, pl.DataFrame):
            base_lf = base_lf.lazy()
        base_lf.join(
            out_lf,
            on=self.cfg.id_col,
            how="left",
            maintain_order="left",
        ).sink_parquet(output_path)
        return True

    async def run(
        self,
        job: SwarmJob,
        data: pl.DataFrame | pl.LazyFrame,
        *,
        input_col: str,
        output_cols: list[str] | None,
        output_mode: OutputJoinMode | None = None,
        output_path: Path | None = None,
    ) -> pl.DataFrame | None:
        """Run a SwarmJob against a polars DataFrame/LazyFrame using batch iteration.

        Args:
            job: SwarmJob instance to execute.
            data: Input polars DataFrame or LazyFrame.
            input_col: Column name to read inputs from.
            output_cols: Output column names (None for dict outputs).
            output_mode: Output join mode (APPEND, REPLACE, IO_ONLY).
            output_path: Optional directory output path for direct shard writes.

        Returns:
            Polars DataFrame with job outputs (and inputs depending on mode), or None if
            the output is written directly to a directory.
        """
        import polars as pl

        if input_col not in self._column_names(data):
            raise ValueError(f"input_col '{input_col}' not found in dataset")

        mode = output_mode or getattr(job, "output_mode", OutputJoinMode.APPEND)
        if isinstance(mode, str):
            mode = OutputJoinMode(mode)

        data = self._ensure_id(data)

        # Load done ids from existing checkpoint shards. We do this by calling prepare() on an empty
        # table to populate store.done_ids without requiring full input conversion to Arrow.
        empty = pa.Table.from_pydict({self.cfg.id_col: []})
        _ = self.store.prepare(empty, self.cfg.id_col)
        done_ids: set[Any] = set(getattr(self.store, "done_ids", set()))

        batch_size = (
            self.cfg.batch_size
            or getattr(job, "native_batch_size", None)
            or self.cfg.checkpoint_every
        )

        for job_batch in self.backend.iter_job_batches(
            data,
            batch_size=int(batch_size),
            id_col=self.cfg.id_col,
            input_col=input_col,
        ):
            batch = job_batch.batch
            if not isinstance(batch, pl.DataFrame):
                raise ValueError("Polars runner expects job batches to carry polars.DataFrame")

            ids = job_batch.ids
            items = job_batch.items

            todo_indices = [i for i, item_id in enumerate(ids) if item_id not in done_ids]
            if not todo_indices:
                continue

            todo_ids = [ids[i] for i in todo_indices]
            todo_items = [items[i] for i in todo_indices]

            async def _on_flush(
                local_indices: list[int],
                local_outputs: list[Any],
                *,
                _todo_ids: list[Any] = todo_ids,
            ) -> None:
                flush_ids = [_todo_ids[i] for i in local_indices]
                rows, cols = _normalize_batch_outputs(local_outputs, output_cols)
                await self.store.flush(
                    FlushBatch(ids=flush_ids, rows=rows),
                    output_cols=cols,
                )
                done_ids.update(flush_ids)

            await job.transform_streaming(
                todo_items,
                on_flush=_on_flush,
                checkpoint_every=self.cfg.checkpoint_every,
            )

        if (
            output_path is not None
            and (output_path.is_dir() or output_path.suffix == "")
            and isinstance(self.store, ArrowShardStore)
            and self._stream_output_to_dir(
                data=data,
                input_col=input_col,
                mode=mode,
                output_path=output_path,
            )
        ):
            return None

        out_table = await asyncio.to_thread(self.store.finalize)
        out_df = cast(pl.DataFrame, pl.from_arrow(out_table))

        if mode == OutputJoinMode.REPLACE:
            return out_df

        if isinstance(data, pl.LazyFrame):
            base_lf = (
                data.select([self.cfg.id_col, input_col])
                if mode == OutputJoinMode.IO_ONLY
                else data
            )
            return base_lf.join(
                out_df.lazy(),
                on=self.cfg.id_col,
                how="left",
                maintain_order="left",
            ).collect(engine="streaming")

        base_df = (
            data.select([self.cfg.id_col, input_col]) if mode == OutputJoinMode.IO_ONLY else data
        )
        return base_df.join(out_df, on=self.cfg.id_col, how="left", maintain_order="left")


async def run_polars_job(
    job_factory: Any,
    backend: DataBackend,
    data: Any,
    *,
    input_col: str,
    output_cols: list[str] | None,
    store_uri: str | None,
    checkpoint_every: int,
    checkpointing: bool,
    id_col: str,
    output_path: Path | None = None,
) -> Any:
    """Run a job using the polars runner with Arrow-based checkpointing.

    Args:
        job_factory: Callable returning a SwarmJob instance.
        backend: Data backend (must yield polars batches for iter_batches).
        data: Input data (polars DataFrame or LazyFrame).
        input_col: Input column name.
        output_cols: Output column names (None for dict outputs).
        store_uri: Base checkpoint store URI (required if checkpointing).
        checkpoint_every: Flush interval in items.
        checkpointing: Whether to read/write checkpoint state.
        id_col: Column name used for stable row ids.
        output_path: Optional output path for direct shard writes.

    Returns:
        Backend-native output (polars DataFrame) or None when outputs are written directly
        to a directory.
    """
    if checkpointing and store_uri is None:
        raise ValueError("store_uri is required when checkpointing is enabled.")

    try:
        import polars as pl  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Polars runner requires `polars` to be installed.") from exc

    store = ArrowShardStore(store_uri) if checkpointing and store_uri else InMemoryArrowStore()
    cfg = PolarsRunnerConfig(id_col=id_col, checkpoint_every=checkpoint_every)
    runner = PolarsJobRunner(store, backend, cfg)
    job = job_factory()
    return await runner.run(
        job,
        data,
        input_col=input_col,
        output_cols=output_cols,
        output_mode=getattr(job, "output_mode", OutputJoinMode.APPEND),
        output_path=output_path,
    )
