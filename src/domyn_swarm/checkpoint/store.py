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

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Optional, Protocol

import pandas as pd

from domyn_swarm.helpers.logger import setup_logger

try:
    import fsspec  # type: ignore
except Exception as _e:  # pragma: no cover
    fsspec = None  # noqa: F401

logger = setup_logger("domyn_swarm.checkpoint.store", level=logging.INFO)


@dataclass
class FlushBatch:
    """A unit of checkpoint persistence.

    ids: stable keys for each output row (e.g., original row ids)
    rows: outputs for each id (scalar, tuple, or dict)
    """

    ids: list[Any]
    rows: list[Any]


class CheckpointStore(Protocol):
    def prepare(self, df: pd.DataFrame, id_col: str) -> pd.DataFrame: ...

    async def flush(
        self, batch: FlushBatch, output_cols: Optional[list[str]]
    ) -> None: ...

    def finalize(self) -> pd.DataFrame: ...


class ParquetShardStore(CheckpointStore):
    """Parquet shard-based checkpointing that works with local or cloud URIs.

    Examples
    --------
    >>> store = ParquetShardStore("s3://bucket/checkpoints/run.parquet")
    >>> todo = store.prepare(df, id_col="_row_id")
    >>> await store.flush(FlushBatch(ids=[...], rows=[...] ), output_cols=["result"])  # many times
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
                "Install fsspec and relevant filesystem extras (s3fs, gcsfs, adlfs, ...) to use ParquetShardStore"
            )
        self.fs, _ = fsspec.core.url_to_fs(self.dir_uri)
        self.fs.mkdirs(self.dir_uri, exist_ok=True)
        self.id_col = "_row_id"
        self.done_ids: set[Any] = set()

    def prepare(self, df: pd.DataFrame, id_col: str) -> pd.DataFrame:
        df = df.copy()
        self.id_col = id_col
        if self.fs.exists(self.base_uri):
            done = pd.read_parquet(
                self.base_uri, storage_options=self.fs.storage_options
            )
            self.done_ids = set(
                done.index.tolist()
                if self.id_col in done.index.names
                else done[self.id_col].tolist()
            )
        mask = ~df[id_col].isin(list(self.done_ids))
        return df.loc[mask]

    async def flush(self, batch: FlushBatch, output_cols: Optional[list[str]]) -> None:
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
        tmp = pd.DataFrame({self.id_col: batch.ids})
        if output_cols is None:
            # assume dict outputs
            tmp = tmp.join(pd.DataFrame(batch.rows))
        else:
            if len(output_cols) == 1:
                tmp[output_cols[0]] = batch.rows
            else:
                logger.info(f"Output columns are: {output_cols}")
                for i, c in enumerate(output_cols):
                    if isinstance(batch.rows[0], (list, tuple)):
                        tmp[c] = [r[i] for r in batch.rows]
                    elif isinstance(batch.rows[0], dict):
                        tmp[c] = [r[c] for r in batch.rows]
                    else:
                        raise ValueError(
                            "When multiple output columns are specified, each row must be a tuple or dict"
                        )
        part = self.dir_uri + f"part-{uuid.uuid4().hex}.parquet"
        tmp = tmp.set_index(self.id_col, drop=True)
        tmp.to_parquet(part, storage_options=self.fs.storage_options)

    def finalize(self) -> pd.DataFrame:
        """
        Merge all shards and write the final parquet file.

        Returns the final merged DataFrame.
        """
        parts = sorted(self.fs.glob(self.dir_uri + "part-*.parquet"))
        if not parts:
            # Return existing merged file if present
            if self.fs.exists(self.base_uri):
                return pd.read_parquet(
                    self.base_uri, storage_options=self.fs.storage_options
                )
            return pd.DataFrame().set_index(self.id_col)
        dfs: list[pd.DataFrame] = [
            pd.read_parquet(p, storage_options=self.fs.storage_options) for p in parts
        ]
        out: pd.DataFrame = pd.concat(dfs, axis=0)
        out = out.loc[~out.index.duplicated(keep="last"), :].sort_index()
        out.to_parquet(self.base_uri, storage_options=self.fs.storage_options)
        return out
