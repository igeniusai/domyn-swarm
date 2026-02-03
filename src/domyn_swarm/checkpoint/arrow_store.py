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
import logging
from typing import Any

import fsspec
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from ulid import ULID

from domyn_swarm.checkpoint.store import CheckpointStore, FlushBatch
from domyn_swarm.helpers.logger import setup_logger

logger = setup_logger("domyn_swarm.checkpoint.arrow_store", level=logging.INFO)

_OFFSET_OVERFLOW_HINT = "offset overflow while concatenating arrays"


def _promote_variable_width_to_large(table: pa.Table) -> pa.Table:
    """Promote variable-width columns to their 64-bit offset equivalents.

    This avoids failures like:
      "offset overflow while concatenating arrays, consider casting input from `string`
      to `large_string` first."

    Args:
        table: Input Arrow table.

    Returns:
        The input table with `string`/`binary` columns cast to `large_string`/`large_binary`.
    """
    schema = table.schema
    updated_fields: list[pa.Field] = []
    changed = False

    for field in schema:
        dtype = field.type

        if pa.types.is_string(dtype) and not pa.types.is_large_string(dtype):
            updated_fields.append(field.with_type(pa.large_string()))
            changed = True
            continue
        if pa.types.is_binary(dtype) and not pa.types.is_large_binary(dtype):
            updated_fields.append(field.with_type(pa.large_binary()))
            changed = True
            continue
        if pa.types.is_dictionary(dtype) and dtype.value_type is not None:
            value_type = dtype.value_type
            if pa.types.is_string(value_type) and not pa.types.is_large_string(value_type):
                updated_fields.append(
                    field.with_type(pa.dictionary(dtype.index_type, pa.large_string()))
                )
                changed = True
                continue
            if pa.types.is_binary(value_type) and not pa.types.is_large_binary(value_type):
                updated_fields.append(
                    field.with_type(pa.dictionary(dtype.index_type, pa.large_binary()))
                )
                changed = True
                continue

        updated_fields.append(field)

    if not changed:
        return table

    return table.cast(pa.schema(updated_fields, metadata=schema.metadata), safe=False)


def _take_with_offset_overflow_fallback(table: pa.Table, indices: list[int]) -> pa.Table:
    """Take rows from a table, retrying with large offsets on overflow.

    Args:
        table: Input Arrow table.
        indices: Row indices to keep.

    Returns:
        Table restricted to the provided indices.
    """
    indices_arr = pa.array(indices, type=pa.int64())
    try:
        out = pc.take(table, indices_arr)  # type: ignore[arg-type]
        if isinstance(out, pa.Table):
            return out
        raise TypeError(f"Expected pc.take(Table, ...) -> Table, got {type(out)!r}") from None
    except pa.ArrowInvalid as e:
        if _OFFSET_OVERFLOW_HINT not in str(e):
            raise
        promoted = _promote_variable_width_to_large(table)
        if promoted is table:
            raise
        logger.warning(
            "Arrow take failed with offset overflow; promoting string/binary columns to "
            "large_* types and retrying."
        )
        out = pc.take(promoted, indices_arr)  # type: ignore[arg-type]
        if isinstance(out, pa.Table):
            return out
        raise TypeError(f"Expected pc.take(Table, ...) -> Table, got {type(out)!r}") from None


def _normalize_tables_for_concat(tables: list[pa.Table], id_col: str) -> list[pa.Table]:
    """Normalize tables so `pa.concat_tables()` can merge them.

    In practice, this resolves common mismatches like `string` vs `large_string` (or
    `binary` vs `large_binary`) by promoting to the 64-bit offset variants.

    Args:
        tables: Input tables to concatenate.
        id_col: Column name for row ids.

    Returns:
        Normalized tables.
    """
    normalized: list[pa.Table] = []
    for t in tables:
        if id_col not in t.column_names:
            for candidate in ("__index_level_0__", "index", "level_0"):
                if candidate in t.column_names:
                    t = t.rename_columns([id_col if c == candidate else c for c in t.column_names])
                    break
        normalized.append(_promote_variable_width_to_large(t))
    return normalized


@dataclass
class InMemoryArrowStore(CheckpointStore[pa.Table]):
    """In-memory checkpoint store for Arrow tables."""

    id_col: str = "_row_id"
    _rows_by_id: dict[Any, dict[str, Any]] | None = None

    def prepare(self, data: pa.Table, id_col: str) -> pa.Table:
        """Prepare a table for processing (no-op for in-memory store).

        Args:
            table: Input Arrow table.
            id_col: Column name for row ids.

        Returns:
            The input table unchanged.
        """
        self.id_col = id_col
        self._rows_by_id = {}
        return data

    async def flush(self, batch: FlushBatch, output_cols: list[str] | None) -> None:
        """Store a batch of outputs in memory.

        Args:
            batch: Batch of ids and output rows.
            output_cols: Output column names (None for dict outputs).
        """
        if self._rows_by_id is None:
            self._rows_by_id = {}
        if output_cols is None:
            for item_id, row in zip(batch.ids, batch.rows):
                if not isinstance(row, dict):
                    raise ValueError("output_cols=None requires dict outputs per row")
                self._rows_by_id[item_id] = dict(row)
            return
        if len(output_cols) == 1:
            col = output_cols[0]
            for item_id, row in zip(batch.ids, batch.rows):
                self._rows_by_id[item_id] = {col: row}
            return
        for item_id, row in zip(batch.ids, batch.rows):
            if isinstance(row, list | tuple):
                self._rows_by_id[item_id] = {c: row[i] for i, c in enumerate(output_cols)}
            elif isinstance(row, dict):
                self._rows_by_id[item_id] = {c: row.get(c) for c in output_cols}
            else:
                raise ValueError(
                    "When multiple output columns are specified, each row must be a tuple or dict"
                )

    def finalize(self) -> pa.Table:
        """Finalize and return an Arrow table of all in-memory outputs.

        Returns:
            Arrow table containing accumulated outputs.
        """
        if not self._rows_by_id:
            return pa.Table.from_pydict({self.id_col: []})
        rows = [{self.id_col: item_id, **payload} for item_id, payload in self._rows_by_id.items()]
        return pa.Table.from_pylist(rows)


class ArrowShardStore(CheckpointStore[pa.Table]):
    """Arrow-native shard store. Mirrors ParquetShardStore but returns Arrow tables."""

    def __init__(self, base_uri: str):
        self.base_uri = base_uri
        if ".parquet" in base_uri:
            self.dir_uri = base_uri.rsplit(".", 1)[0] + "/"
        else:
            self.dir_uri = base_uri.rstrip("/") + "/"
            self.base_uri = self.dir_uri.rstrip("/") + ".parquet"
        self.fs, self.dir_path = fsspec.core.url_to_fs(self.dir_uri)
        if not self.dir_path.endswith("/"):
            self.dir_path = self.dir_path + "/"
        self.base_path = self.dir_path.rstrip("/") + ".parquet"
        self.fs.mkdirs(self.dir_path, exist_ok=True)
        self.id_col = "_row_id"
        self.done_ids: set[Any] = set()

    def _normalize_id_column(self, table: pa.Table) -> pa.Table:
        """Ensure the id column is named `self.id_col`.

        Args:
            table: Input Arrow table.

        Returns:
            Table with the id column normalized.
        """
        if self.id_col in table.column_names:
            return table
        for candidate in ("__index_level_0__", "index", "level_0"):
            if candidate in table.column_names:
                return table.rename_columns(
                    [self.id_col if c == candidate else c for c in table.column_names]
                )
        raise ValueError(f"Merged parquet is missing id column {self.id_col!r}")

    def prepare(self, data: pa.Table, id_col: str) -> pa.Table:
        """Filter the input table to rows not yet in checkpoints.

        Args:
            table: Input Arrow table.
            id_col: Column name for row ids.

        Returns:
            Filtered Arrow table containing only rows to process.
        """
        self.id_col = id_col
        done_ids: set[Any] = set()

        def _read_ids(path: str) -> set[Any]:
            with self.fs.open(path, "rb") as f:
                pf = pq.ParquetFile(f)
                cols = pf.schema_arrow.names
                col = self.id_col if self.id_col in cols else None
                if col is None:
                    for candidate in ("__index_level_0__", "index", "level_0"):
                        if candidate in cols:
                            col = candidate
                            break
                if col is None:
                    raise ValueError(f"Parquet file missing id column {self.id_col!r}: {path}")
                t = pf.read(columns=[col])
            return set(t.column(0).to_pylist())

        if self.fs.exists(self.base_path):
            done_ids.update(_read_ids(self.base_path))
        if self.fs.exists(self.dir_path):
            for path in self.fs.glob(self.dir_path + "part-*.parquet"):
                done_ids.update(_read_ids(path))

        self.done_ids = done_ids
        if not done_ids:
            return data

        col = data[self.id_col] if self.id_col in data.column_names else None
        if col is None:
            raise ValueError(f"Input table missing id column {self.id_col!r}")
        mask = pc.invert(pc.is_in(col, value_set=pa.array(list(done_ids))))  # type: ignore
        return data.filter(mask)

    async def flush(self, batch: FlushBatch, output_cols: list[str] | None) -> None:
        """Write a batch of outputs to a shard parquet file.

        Args:
            batch: Batch of ids and output rows.
            output_cols: Output column names (None for dict outputs).
        """
        await asyncio.to_thread(self._flush_sync, batch, output_cols)

    def _flush_sync(self, batch: FlushBatch, output_cols: list[str] | None) -> None:
        """Synchronously write a batch of outputs to a parquet shard.

        Args:
            batch: Batch of ids and output rows.
            output_cols: Output column names (None for dict outputs).
        """
        out: dict[str, list[Any]] = {self.id_col: list(batch.ids)}
        if output_cols is None:
            if batch.rows and isinstance(batch.rows[0], dict):
                keys = set().union(*(r.keys() for r in batch.rows))
                for k in sorted(keys):
                    out[k] = [r.get(k) for r in batch.rows]
            else:
                raise ValueError("output_cols=None requires dict outputs per row")
        elif len(output_cols) == 1:
            out[output_cols[0]] = list(batch.rows)
        else:
            first = batch.rows[0] if batch.rows else None
            if isinstance(first, list | tuple):
                for i, c in enumerate(output_cols):
                    out[c] = [r[i] for r in batch.rows]
            elif isinstance(first, dict):
                for c in output_cols:
                    out[c] = [r.get(c) for r in batch.rows]
            else:
                raise ValueError(
                    "When multiple output columns are specified, each row must be a tuple or dict"
                )

        part = self.dir_path + f"part-{str(ULID()).lower()}.parquet"
        table = pa.Table.from_pydict(out)
        with self.fs.open(part, "wb") as f:
            pq.write_table(table, f, use_dictionary=False)

    def finalize(self) -> pa.Table:
        """Merge shard parts, persist the merged parquet, and return an Arrow table.

        Returns:
            Arrow table containing merged outputs.
        """
        parts = sorted(self.fs.glob(self.dir_path + "part-*.parquet"))
        if not parts:
            if self.fs.exists(self.base_path):
                with self.fs.open(self.base_path, "rb") as f:
                    table = pq.read_table(f)
                return self._normalize_id_column(table)
            return pa.Table.from_pydict({self.id_col: []})

        tables: list[pa.Table] = []
        if self.fs.exists(self.base_path):
            with self.fs.open(self.base_path, "rb") as f:
                base_table = pq.read_table(f)
            tables.append(self._normalize_id_column(base_table))
        for p in parts:
            with self.fs.open(p, "rb") as f:
                tables.append(pq.read_table(f))

        tables = [self._normalize_id_column(t) for t in tables]
        tables = _normalize_tables_for_concat(tables, self.id_col)
        table = pa.concat_tables(tables, promote_options="default")
        table = self._normalize_id_column(table)
        ids = table.column(self.id_col).to_pylist()
        last: dict[Any, int] = {v: i for i, v in enumerate(ids)}
        keep_indices = sorted(last.values())
        table = _take_with_offset_overflow_fallback(table, keep_indices)

        with self.fs.open(self.base_path, "wb") as f:
            pq.write_table(table, f, use_dictionary=False)

        return table
