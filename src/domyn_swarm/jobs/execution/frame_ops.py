# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Frame-type-specific operations behind the shared execution pipeline.

`run_sharded_pipeline` owns the control flow shared by every eager execution
engine; everything that depends on the concrete frame type (pandas DataFrame,
Arrow table) lives behind the `FrameOps` protocol implemented here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from domyn_swarm.checkpoint.arrow_store import ArrowShardStore
from domyn_swarm.checkpoint.store import CheckpointStore, ParquetShardStore
from domyn_swarm.data.backends.base import DataBackend
from domyn_swarm.jobs.api.base import OutputJoinMode
from domyn_swarm.jobs.api.runner import JobRunner, RunnerConfig
from domyn_swarm.jobs.io.sharding import shard_indices_by_id

T = TypeVar("T")


@dataclass(frozen=True)
class ShardSpec:
    """Per-run settings that every shard of a job shares.

    Attributes:
        input_col: Column holding the items passed to the job.
        output_cols: Output column names, or None for dict-valued outputs.
        id_col: Column holding stable row ids used for checkpoint resume.
        checkpoint_every: Flush interval, in items.
        checkpointing: Whether checkpoint state is read and written.
        output_mode: How outputs are joined back onto the input frame.
    """

    input_col: str
    output_cols: list[str] | None
    id_col: str
    checkpoint_every: int
    checkpointing: bool
    output_mode: OutputJoinMode


class FrameOps(Protocol[T]):
    """Operations the shared pipeline needs from a concrete frame type."""

    def coerce(self, data: Any, id_col: str) -> T:
        """Convert backend-native input into this adapter's frame type.

        Args:
            data: Backend-native data, or a frame already of this adapter's type.
            id_col: Column name used for stable row ids. Adapters that would
                otherwise lose the row identity during conversion (e.g. Arrow,
                whose backend conversion drops the pandas index) use this to
                preseed the id column before converting.
        """
        ...

    def column_names(self, frame: T) -> list[str]:
        """Return the frame's column names."""
        ...

    def ensure_id(self, frame: T, id_col: str) -> T:
        """Return a frame guaranteed to carry `id_col`."""
        ...

    def filter_out_ids(self, frame: T, id_col: str, done: set[Any]) -> T:
        """Drop rows whose id appears in `done`."""
        ...

    def shard_indices(self, frame: T, id_col: str, mode: str, n: int) -> list[np.ndarray]:
        """Partition row positions into `n` shards."""
        ...

    def take(self, frame: T, indices: np.ndarray) -> T:
        """Select rows by label, as produced by ``shard_mode="index"``."""
        ...

    def take_positional(self, frame: T, indices: np.ndarray) -> T:
        """Select rows by position, as produced by ``shard_mode="id"``.

        Identical to :meth:`take` for frame types with no separate row labels.
        """
        ...

    def concat(self, parts: list[T]) -> T:
        """Combine shard outputs back into one frame."""
        ...

    async def run_shard(
        self, job_factory: Callable[[], Any], frame: T, store_uri: str | None, spec: ShardSpec
    ) -> T:
        """Execute one shard and return its outputs."""
        ...

    def empty_with_id(self, id_col: str) -> T:
        """Return an empty frame carrying only `id_col`, for resume bootstrapping."""
        ...

    @property
    def store_factory(self) -> Callable[[str], CheckpointStore]:
        """The checkpoint store class this frame type reads and writes."""
        ...


class PandasFrameOps:
    """`FrameOps` over `pandas.DataFrame`."""

    def __init__(self, backend: DataBackend):
        self.backend = backend

    def coerce(self, data: Any, id_col: str) -> pd.DataFrame:
        """Convert backend-native input into a DataFrame.

        `id_col` is accepted for parity with the `FrameOps` protocol but
        unused: a pandas frame keeps its index through conversion, and
        `ensure_id` derives missing ids from that index later.
        """
        return data if isinstance(data, pd.DataFrame) else self.backend.to_pandas(data)

    def column_names(self, frame: pd.DataFrame) -> list[str]:
        """Return the frame's column names."""
        return list(frame.columns)

    def ensure_id(self, frame: pd.DataFrame, id_col: str) -> pd.DataFrame:
        """Derive missing ids from the frame index."""
        if id_col in frame.columns:
            return frame
        frame = frame.copy(deep=False)
        frame[id_col] = frame.index
        return frame

    def filter_out_ids(self, frame: pd.DataFrame, id_col: str, done: set[Any]) -> pd.DataFrame:
        """Drop rows whose id is already done."""
        if not done:
            return frame
        return frame.loc[~frame[id_col].isin(list(done))]

    def shard_indices(
        self, frame: pd.DataFrame, id_col: str, mode: str, n: int
    ) -> list[np.ndarray]:
        """Partition row positions into `n` shards."""
        if mode == "index":
            return list(np.array_split(frame.index, n))
        return shard_indices_by_id(cast(pd.Series, frame[id_col]), n)

    def take(self, frame: pd.DataFrame, indices: np.ndarray) -> pd.DataFrame:
        """Select rows, matching the sharding mode's index semantics."""
        return frame.loc[indices].copy(deep=False)

    def take_positional(self, frame: pd.DataFrame, indices: np.ndarray) -> pd.DataFrame:
        """Select rows by position rather than label."""
        return frame.iloc[indices].copy(deep=False)

    def concat(self, parts: list[pd.DataFrame]) -> pd.DataFrame:
        """Combine shard outputs, restoring original row order."""
        return pd.concat(parts).sort_index()

    async def run_shard(
        self,
        job_factory: Callable[[], Any],
        frame: pd.DataFrame,
        store_uri: str | None,
        spec: ShardSpec,
    ) -> pd.DataFrame:
        """Execute one shard through `JobRunner`."""
        from domyn_swarm.jobs.io.checkpointing import _build_checkpoint_store

        store = _build_checkpoint_store(checkpointing=spec.checkpointing, store_uri=store_uri)
        cfg = RunnerConfig(id_col=spec.id_col, checkpoint_every=spec.checkpoint_every)
        return await JobRunner(store, cfg).run(
            job_factory(),
            frame,
            input_col=spec.input_col,
            output_cols=spec.output_cols,
            output_mode=spec.output_mode,
        )

    def empty_with_id(self, id_col: str) -> pd.DataFrame:
        """Return an empty DataFrame carrying only `id_col`."""
        return pd.DataFrame({id_col: []})

    @property
    def store_factory(self) -> Callable[[str], CheckpointStore]:
        """Checkpoint store class for pandas frames."""
        return ParquetShardStore


class ArrowFrameOps:
    """`FrameOps` over `pyarrow.Table`."""

    def __init__(self, backend: DataBackend):
        self.backend = backend

    def coerce(self, data: Any, id_col: str) -> pa.Table:
        """Convert backend-native input into an Arrow table.

        A pandas frame's index is preseeded into `id_col` before conversion:
        Arrow backend conversion drops the pandas index (`preserve_index=False`),
        so without this the row identity would be lost and `ensure_id` would
        fall back to a fresh positional range instead of the original ids.
        """
        if isinstance(data, pa.Table):
            return data
        if isinstance(data, pd.DataFrame):
            df = data
            if id_col not in df.columns:
                df = df.copy(deep=False)
                df[id_col] = df.index
            return self.backend.to_arrow(df)
        return self.backend.to_arrow(data)

    def column_names(self, frame: pa.Table) -> list[str]:
        """Return the table's column names."""
        return list(frame.column_names)

    def ensure_id(self, frame: pa.Table, id_col: str) -> pa.Table:
        """Recover ids from a known index column, else assign positional ids.

        Arrow tables written from pandas carry the index under one of several
        conventional names. Recovering it keeps ids stable across a
        pandas-to-Arrow round trip, which positional ids would not.
        """
        if id_col in frame.column_names:
            return frame
        for candidate in ("__index_level_0__", "index", "level_0"):
            if candidate in frame.column_names:
                return frame.rename_columns(
                    [id_col if c == candidate else c for c in frame.column_names]
                )
        return frame.append_column(id_col, pa.array(range(len(frame))))

    def filter_out_ids(self, frame: pa.Table, id_col: str, done: set[Any]) -> pa.Table:
        """Drop rows whose id is already done."""
        if not done:
            return frame
        mask = pc.invert(pc.is_in(frame[id_col], value_set=pa.array(list(done))))  # type: ignore[arg-type]
        return frame.filter(mask)

    def shard_indices(self, frame: pa.Table, id_col: str, mode: str, n: int) -> list[np.ndarray]:
        """Partition row positions into `n` shards."""
        if mode == "index":
            return list(np.array_split(np.arange(frame.num_rows), n))
        return shard_indices_by_id(frame[id_col].to_pylist(), n)

    def take(self, frame: pa.Table, indices: np.ndarray) -> pa.Table:
        """Select rows by position."""
        return frame.take(pa.array(indices, type=pa.int64()))

    def take_positional(self, frame: pa.Table, indices: np.ndarray) -> pa.Table:
        """Select rows by position (identical to `take` for Arrow)."""
        return self.take(frame, indices)

    def concat(self, parts: list[pa.Table]) -> pa.Table:
        """Combine shard outputs positionally."""
        return pa.concat_tables(parts)

    async def run_shard(
        self,
        job_factory: Callable[[], Any],
        frame: pa.Table,
        store_uri: str | None,
        spec: ShardSpec,
    ) -> pa.Table:
        """Execute one shard through the Arrow runner."""
        from domyn_swarm.jobs.execution.arrow import run_arrow_job

        return await run_arrow_job(
            job_factory,
            frame,
            input_col=spec.input_col,
            output_cols=spec.output_cols,
            store_uri=store_uri,
            checkpoint_every=spec.checkpoint_every,
            checkpointing=spec.checkpointing,
            id_col=spec.id_col,
        )

    def empty_with_id(self, id_col: str) -> pa.Table:
        """Return an empty table carrying only `id_col`."""
        return pa.Table.from_pydict({id_col: []})

    @property
    def store_factory(self) -> Callable[[str], CheckpointStore]:
        """Checkpoint store class for Arrow tables."""
        return ArrowShardStore


__all__ = ["ArrowFrameOps", "FrameOps", "PandasFrameOps", "ShardSpec"]
