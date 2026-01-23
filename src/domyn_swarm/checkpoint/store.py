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
from dataclasses import dataclass
import logging
from typing import Any, Protocol, TypeVar

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from ulid import ULID

from domyn_swarm.helpers.logger import setup_logger

try:
    import fsspec  # type: ignore
except Exception as _e:  # pragma: no cover
    fsspec = None

logger = setup_logger("domyn_swarm.checkpoint.store", level=logging.INFO)


@dataclass
class FlushBatch:
    """A unit of checkpoint persistence.

    ids: stable keys for each output row (e.g., original row ids)
    rows: outputs for each id (scalar, tuple, or dict)
    """

    ids: list[Any]
    rows: list[Any]


T = TypeVar("T")


class CheckpointStore(Protocol[T]):
    def prepare(self, data: T, id_col: str) -> T: ...

    async def flush(self, batch: FlushBatch, output_cols: list[str] | None) -> None: ...

    def finalize(self) -> T: ...


class ParquetShardStore(CheckpointStore[pd.DataFrame]):
    """Parquet shard-based checkpointing that works with local or cloud URIs.

    Examples
    --------
    >>> store = ParquetShardStore("s3://bucket/checkpoints/run.parquet")
    >>> todo = store.prepare(df, id_col="_row_id")
    >>> await store.flush(FlushBatch(ids=[...], rows=[...]), output_cols=["result"])  # many times
    >>> out = store.finalize()  # merges shards and writes the final parquet
    """

    def __init__(self, base_uri: str):
        self.base_uri = base_uri
        if ".parquet" in base_uri:
            self.dir_uri = base_uri.rsplit(".", 1)[0] + "/"
        else:
            self.dir_uri = base_uri.rstrip("/") + "/"
            self.base_uri = self.dir_uri.rstrip("/") + ".parquet"
        if fsspec is None:
            raise ImportError(
                "Install fsspec and relevant filesystem extras "
                "(s3fs, gcsfs, adlfs, ...) to use ParquetShardStore"
            )
        self.fs, self.dir_path = fsspec.core.url_to_fs(self.dir_uri)
        # Normalize to a path understood by the FS implementation (no scheme).
        if not self.dir_path.endswith("/"):
            self.dir_path = self.dir_path + "/"
        self.base_path = self.dir_path.rstrip("/") + ".parquet"
        self.fs.mkdirs(self.dir_path, exist_ok=True)
        self.id_col = "_row_id"
        self.done_ids: set[Any] = set()

    def prepare(self, data: pd.DataFrame, id_col: str) -> pd.DataFrame:
        data = data.copy(deep=False)
        self.id_col = id_col
        done_ids: set[Any] = set()

        if self.fs.exists(self.base_path):
            done_ids.update(self._read_done_ids(self.base_path))

        if self.fs.exists(self.dir_path):
            for path in self.fs.glob(self.dir_path + "*.parquet"):
                done_ids.update(self._read_done_ids(path))

        self.done_ids = done_ids

        if not done_ids:
            # nothing to skip
            return data

        mask = ~data[id_col].isin(list(done_ids))
        return data.loc[mask]

    def _read_done_ids(self, path: str) -> set[Any]:
        with self.fs.open(path, "rb") as f:
            pf = pq.ParquetFile(f)
            cols = pf.schema_arrow.names
            col = self.id_col if self.id_col in cols else None
            if col is None:
                # Best-effort compatibility with pandas-written index columns.
                for candidate in ("__index_level_0__", "index", "level_0"):
                    if candidate in cols:
                        col = candidate
                        break
            if col is None:
                raise ValueError(f"Parquet file missing id column {self.id_col!r}: {path}")
            table = pf.read(columns=[col])
        arr = table.column(0).combine_chunks()
        return set(arr.to_pylist())

    def _normalize_id_column(self, table: pa.Table) -> pa.Table:
        """Ensure the id column is named `self.id_col` (compat with pandas index columns)."""
        if self.id_col in table.column_names:
            return table
        for candidate in ("__index_level_0__", "index", "level_0"):
            if candidate in table.column_names:
                return table.rename_columns(
                    [self.id_col if c == candidate else c for c in table.column_names]
                )
        raise ValueError(f"Merged parquet is missing id column {self.id_col!r}")

    async def flush(self, batch: FlushBatch, output_cols: list[str] | None) -> None:
        """Flush a batch of data to a parquet file.

        This method processes a batch of data and writes it to a parquet file in the
        checkpoint directory. It handles different output formats including single values,
        lists/tuples, and dictionaries.

        Args:
            batch (FlushBatch): The batch containing IDs and row data to flush.
            output_cols (Optional[list[str]]): List of output column names. If None,
                assumes batch.rows contains dictionaries that will be joined directly.
                If a single column, treats batch.rows as simple values. If multiple
                columns, expects batch.rows to contain lists/tuples or dictionaries.

        Raises:
            ValueError: When multiple output columns are specified but batch rows
                are neither tuples/lists nor dictionaries.

        Note:
            The resulting parquet file is saved with a UUID-based filename to avoid
            conflicts and uses the configured ID column as the index.
        """
        await asyncio.to_thread(self._flush_sync, batch, output_cols)

    def _flush_sync(self, batch: FlushBatch, output_cols: list[str] | None) -> None:
        """Synchronously write a batch to a parquet shard.

        Args:
            batch: Batch containing ids and output rows.
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
            logger.info(f"Output columns are: {output_cols}")
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

        table = pa.Table.from_pydict(out)
        # Use a time-sortable filename to ensure deterministic merge semantics.
        # This allows finalize() to interpret "last write wins" by lexicographic order.
        part = self.dir_path + f"part-{str(ULID()).lower()}.parquet"
        with self.fs.open(part, "wb") as f:
            pq.write_table(table, f, use_dictionary=False)

    def finalize(self) -> pd.DataFrame:
        """
        Merge all shards and write the final parquet file.

        Returns the final merged DataFrame.
        """
        parts = sorted(self.fs.glob(self.dir_path + "part-*.parquet"))
        if not parts:
            # Return existing merged file if present
            if self.fs.exists(self.base_path):
                with self.fs.open(self.base_path, "rb") as f:
                    table = pq.read_table(f)
                table = self._normalize_id_column(table)
                df = table.to_pandas(ignore_metadata=True).set_index(self.id_col, drop=True)
                return df
            return pd.DataFrame().set_index(self.id_col)

        tables: list[pa.Table] = []
        for p in parts:
            with self.fs.open(p, "rb") as f:
                tables.append(pq.read_table(f))

        table = pa.concat_tables(tables, promote_options="default")
        # De-dup by id_col (best-effort "last write wins" in the concatenation order).
        table = self._normalize_id_column(table)
        ids = table.column(self.id_col).combine_chunks().to_pylist()
        last: dict[Any, int] = {v: i for i, v in enumerate(ids)}
        keep_indices = sorted(last.values())
        table = table.take(pa.array(keep_indices, type=pa.int64()))

        with self.fs.open(self.base_path, "wb") as f:
            pq.write_table(table, f, use_dictionary=False)

        df = table.to_pandas(ignore_metadata=True).set_index(self.id_col, drop=True)
        return df


class InMemoryStore(CheckpointStore[pd.DataFrame]):
    """In-memory checkpoint store (no read/write to disk).

    Intended for debugging and short runs where checkpointing I/O should be bypassed.
    Semantics: "last write wins" per id.
    """

    def __init__(self):
        self.id_col = "_row_id"
        self._rows_by_id: dict[Any, dict[str, Any]] = {}

    def prepare(self, data: pd.DataFrame, id_col: str) -> pd.DataFrame:
        self.id_col = id_col
        return data

    async def flush(self, batch: FlushBatch, output_cols: list[str] | None) -> None:
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

    def finalize(self) -> pd.DataFrame:
        if not self._rows_by_id:
            return pd.DataFrame().set_index(self.id_col)

        rows = [{self.id_col: item_id, **payload} for item_id, payload in self._rows_by_id.items()]
        return pd.DataFrame(rows).set_index(self.id_col, drop=True)
