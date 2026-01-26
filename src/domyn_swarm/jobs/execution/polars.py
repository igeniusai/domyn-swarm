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
from collections.abc import Callable, Iterator
from dataclasses import dataclass
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
import pyarrow as pa

from domyn_swarm.checkpoint.arrow_store import ArrowShardStore, InMemoryArrowStore
from domyn_swarm.checkpoint.store import FlushBatch
from domyn_swarm.data.backends.base import DataBackend
from domyn_swarm.jobs.api.base import OutputJoinMode, SwarmJob
from domyn_swarm.jobs.api.runner import normalize_batch_outputs
from domyn_swarm.jobs.io.checkpointing import (
    _shard_store_uri,
    _validate_checkpoint_store,
    _validate_sharded_execution,
)
from domyn_swarm.jobs.io.columns import _require_column_names, _validate_required_id

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
        # Polars streams to a single parquet file; represent "directory output" as a dataset
        # containing one streamed shard. This keeps memory bounded for large outputs.
        target = output_path / "data-000000.parquet"
        if mode == OutputJoinMode.REPLACE:
            out_lf.sink_parquet(target)
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
        ).sink_parquet(target)
        return True

    def _resolve_mode(
        self,
        job: SwarmJob,
        output_mode: OutputJoinMode | None,
    ) -> OutputJoinMode:
        """Resolve the output mode for a run.

        Args:
            job: SwarmJob instance providing defaults.
            output_mode: Optional output mode override.

        Returns:
            Resolved OutputJoinMode.
        """
        mode = output_mode or getattr(job, "output_mode", OutputJoinMode.APPEND)
        return OutputJoinMode(mode) if isinstance(mode, str) else mode

    def _ensure_input_col(self, data: pl.DataFrame | pl.LazyFrame, input_col: str) -> None:
        """Validate that the input column exists when eager data is used.

        Args:
            data: Polars DataFrame or LazyFrame.
            input_col: Column name to read inputs from.

        Raises:
            ValueError: If the input column is missing on eager data.
        """
        import polars as pl

        if not isinstance(data, pl.LazyFrame) and input_col not in self._column_names(data):
            raise ValueError(f"input_col '{input_col}' not found in dataset")

    def _prepare_done_ids(self) -> set[Any]:
        """Populate checkpoint store state and return known done ids.

        Returns:
            Set of ids already processed.
        """
        empty = pa.Table.from_pydict({self.cfg.id_col: []})
        _ = self.store.prepare(empty, self.cfg.id_col)
        return set(getattr(self.store, "done_ids", set()))

    def _resolve_batch_size(self, job: SwarmJob) -> int:
        """Resolve the batch size for iterating job batches.

        Args:
            job: SwarmJob instance providing defaults.

        Returns:
            Batch size to use.
        """
        return int(
            self.cfg.batch_size
            or getattr(job, "native_batch_size", None)
            or self.cfg.checkpoint_every
        )

    def _iter_todo_batches(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        *,
        input_col: str,
        done_ids: set[Any],
        batch_size: int,
    ) -> Iterator[tuple[pl.DataFrame, list[Any], list[Any]]]:
        """Iterate batches that still need processing.

        Args:
            data: Polars DataFrame or LazyFrame.
            input_col: Column name to read inputs from.
            done_ids: Set of ids already processed.
            batch_size: Batch size for iteration.

        Yields:
            Tuples of (batch, todo_ids, todo_items).
        """
        import polars as pl

        for job_batch in self.backend.iter_job_batches(
            data,
            batch_size=batch_size,
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
            yield batch, todo_ids, todo_items

    def _build_flush_cb(
        self,
        todo_ids: list[Any],
        output_cols: list[str] | None,
        done_ids: set[Any],
    ):
        """Build a flush callback that writes outputs and updates done ids.

        Args:
            todo_ids: IDs for the current batch of work.
            output_cols: Output column names (None for dict outputs).
            done_ids: Mutable set of processed ids.

        Returns:
            Async callback suitable for SwarmJob.transform_streaming.
        """

        async def _on_flush(local_indices: list[int], local_outputs: list[Any]) -> None:
            flush_ids = [todo_ids[i] for i in local_indices]
            rows, cols = normalize_batch_outputs(local_outputs, output_cols)
            await self.store.flush(
                FlushBatch(ids=flush_ids, rows=rows),
                output_cols=cols,
            )
            done_ids.update(flush_ids)

        return _on_flush

    def _finalize_output(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        *,
        out_df: pl.DataFrame,
        mode: OutputJoinMode,
        input_col: str,
    ) -> pl.DataFrame:
        """Finalize and join outputs for a run.

        Args:
            data: Polars DataFrame or LazyFrame.
            out_df: Output DataFrame built from checkpoint store.
            mode: Output join mode.
            input_col: Input column name.

        Returns:
            Final Polars DataFrame with outputs joined.
        """
        import polars as pl

        if mode == OutputJoinMode.REPLACE:
            return out_df

        if isinstance(data, pl.LazyFrame):
            base_lf = (
                data.select([self.cfg.id_col, input_col])
                if mode == OutputJoinMode.IO_ONLY
                else data
            )
            return base_lf.join(
                out_df.lazy(), on=self.cfg.id_col, how="left", maintain_order="left"
            ).collect()

        base_df = (
            data.select([self.cfg.id_col, input_col]) if mode == OutputJoinMode.IO_ONLY else data
        )
        return base_df.join(out_df, on=self.cfg.id_col, how="left", maintain_order="left")

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

        self._ensure_input_col(data, input_col)

        mode = self._resolve_mode(job, output_mode)
        data = self._ensure_id(data)

        done_ids = self._prepare_done_ids()
        batch_size = self._resolve_batch_size(job)

        for _batch, todo_ids, todo_items in self._iter_todo_batches(
            data,
            input_col=input_col,
            done_ids=done_ids,
            batch_size=batch_size,
        ):
            on_flush = self._build_flush_cb(todo_ids, output_cols, done_ids)
            await job.transform_streaming(
                todo_items,
                on_flush=on_flush,
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
        return self._finalize_output(
            data,
            out_df=out_df,
            mode=mode,
            input_col=input_col,
        )


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
    _validate_checkpoint_store(checkpointing, store_uri)

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


def _coerce_polars_data(data: Any) -> Any:
    """Convert supported inputs into a polars DataFrame or LazyFrame.

    Args:
        data: Input dataset-like object.

    Returns:
        Polars DataFrame or LazyFrame.

    Raises:
        ValueError: If the input cannot be converted to polars.
    """
    import polars as pl

    if isinstance(data, pd.DataFrame):
        data = pl.from_pandas(data)
    elif isinstance(data, pa.Table):
        data = pl.from_arrow(data)

    if not isinstance(data, pl.DataFrame | pl.LazyFrame):
        raise ValueError("Polars runner expects a polars DataFrame or LazyFrame.")
    return data


def _validate_polars_inputs(
    data: Any,
    *,
    require_id: bool,
    id_col: str,
    checkpointing: bool,
    store_uri: str | None,
) -> None:
    """Validate polars execution preconditions.

    Args:
        data: Polars DataFrame or LazyFrame.
        require_id: Whether id_col must already exist in the input.
        id_col: Column name used for stable row ids.
        checkpointing: Whether checkpointing is enabled.
        store_uri: Store URI for checkpointing.

    Raises:
        ValueError: If required preconditions are not met.
    """
    if require_id:
        import polars as pl

        if isinstance(data, pl.LazyFrame):
            if id_col not in data.collect_schema().names():
                raise ValueError(f"Missing required id column: {id_col}")
        else:
            _validate_required_id(data, id_col)
    _validate_checkpoint_store(checkpointing, store_uri)


def _ensure_polars_id(data: Any, id_col: str) -> Any:
    """Ensure the id column exists on a polars frame.

    Args:
        data: Polars DataFrame or LazyFrame.
        id_col: Column name used for stable row ids.

    Returns:
        Polars DataFrame or LazyFrame with id_col present.
    """
    import polars as pl

    if isinstance(data, pl.LazyFrame):
        names = data.collect_schema().names()
    else:
        names = _require_column_names(data)
    if id_col not in names:
        return data.with_row_index(id_col)
    return data


def _collect_polars_data(data: Any) -> Any:
    """Collect a polars LazyFrame into a DataFrame.

    Args:
        data: Polars DataFrame or LazyFrame.

    Returns:
        Polars DataFrame.
    """
    import polars as pl

    if isinstance(data, pl.LazyFrame):
        return data.collect()
    return data


async def _run_polars(
    *,
    job_factory: Callable[[], Any],
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
    """Run the polars execution path for non-ray backends.

    Args:
        job_factory: Callable producing a SwarmJob instance.
        backend: Data backend used for conversion.
        data: Polars DataFrame or LazyFrame.
        input_col: Input column name in the dataset.
        output_cols: Output columns to produce (None for dict outputs).
        id_col: Column name used for stable row ids.
        require_id: Whether id_col must already exist in the input.
        nshards: Number of shards to split the input into.
        store_uri: Base checkpoint store URI.
        checkpoint_every: Flush interval in items.
        checkpointing: Whether checkpointing is enabled.
        output_path: Optional output path for direct shard writes.

    Returns:
        Polars DataFrame containing job outputs, or None when outputs are written directly
        to a directory.
    """
    data = _coerce_polars_data(data)
    _validate_polars_inputs(
        data,
        require_id=require_id,
        id_col=id_col,
        checkpointing=checkpointing,
        store_uri=store_uri,
    )

    if nshards <= 1:
        if not require_id:
            data = _ensure_polars_id(data, id_col)
        return await _run_polars_single(
            job_factory=job_factory,
            backend=backend,
            data=data,
            input_col=input_col,
            output_cols=output_cols,
            store_uri=store_uri,
            checkpoint_every=checkpoint_every,
            checkpointing=checkpointing,
            id_col=id_col,
            output_path=output_path,
        )

    _validate_sharded_execution(checkpointing)
    data = _ensure_polars_id(data, id_col)
    data = _collect_polars_data(data)
    return await _run_polars_sharded(
        job_factory=job_factory,
        backend=backend,
        data=data,
        input_col=input_col,
        output_cols=output_cols,
        store_uri=store_uri,
        checkpoint_every=checkpoint_every,
        checkpointing=checkpointing,
        id_col=id_col,
        nshards=nshards,
    )


async def _run_polars_single(
    *,
    job_factory: Callable[[], Any],
    backend: DataBackend,
    data: Any,
    input_col: str,
    output_cols: list[str] | None,
    store_uri: str | None,
    checkpoint_every: int,
    checkpointing: bool,
    id_col: str,
    output_path: Path | None,
) -> Any:
    """Run a single-shard polars job.

    Args:
        job_factory: Callable producing a SwarmJob instance.
        backend: Data backend used for conversion.
        data: Polars DataFrame or LazyFrame.
        input_col: Input column name in the dataset.
        output_cols: Output columns to produce (None for dict outputs).
        store_uri: Base checkpoint store URI.
        checkpoint_every: Flush interval in items.
        checkpointing: Whether checkpointing is enabled.
        id_col: Column name used for stable row ids.
        output_path: Optional output path for direct shard writes.

    Returns:
        Polars DataFrame containing job outputs, or None when outputs are written directly
        to a directory.
    """
    return await run_polars_job(
        job_factory,
        backend,
        data,
        input_col=input_col,
        output_cols=output_cols,
        store_uri=store_uri,
        checkpoint_every=checkpoint_every,
        checkpointing=checkpointing,
        id_col=id_col,
        output_path=output_path,
    )


async def _run_polars_sharded(
    *,
    job_factory: Callable[[], Any],
    backend: DataBackend,
    data: Any,
    input_col: str,
    output_cols: list[str] | None,
    store_uri: str | None,
    checkpoint_every: int,
    checkpointing: bool,
    id_col: str,
    nshards: int,
) -> Any:
    """Run a sharded polars job with checkpointing.

    Args:
        job_factory: Callable producing a SwarmJob instance.
        backend: Data backend used for conversion.
        data: Polars DataFrame.
        input_col: Input column name in the dataset.
        output_cols: Output columns to produce (None for dict outputs).
        store_uri: Base checkpoint store URI.
        checkpoint_every: Flush interval in items.
        checkpointing: Whether checkpointing is enabled.
        id_col: Column name used for stable row ids.
        nshards: Number of shards to split the input into.

    Returns:
        Polars DataFrame containing job outputs.
    """
    import polars as pl

    total_rows = data.height
    if total_rows == 0:
        return await run_polars_job(
            job_factory,
            backend,
            data,
            input_col=input_col,
            output_cols=output_cols,
            store_uri=store_uri,
            checkpoint_every=checkpoint_every,
            checkpointing=checkpointing,
            id_col=id_col,
            output_path=None,
        )

    chunk_size = math.ceil(total_rows / nshards)

    async def _one(i: int, start: int):
        assert store_uri is not None
        sub = data.slice(start, chunk_size)
        su = _shard_store_uri(store_uri, i)
        return await run_polars_job(
            job_factory,
            backend,
            sub,
            input_col=input_col,
            output_cols=output_cols,
            store_uri=su,
            checkpoint_every=checkpoint_every,
            checkpointing=checkpointing,
            id_col=id_col,
            output_path=None,
        )

    tasks = []
    for shard_id in range(nshards):
        start = shard_id * chunk_size
        if start >= total_rows:
            break
        tasks.append(_one(shard_id, start))
    parts = await asyncio.gather(*tasks)
    return pl.concat(parts, how="vertical")
