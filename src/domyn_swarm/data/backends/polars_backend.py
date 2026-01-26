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

from collections.abc import Iterable
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa

from domyn_swarm.data.backends.base import BackendError, DataBackend, JobBatch
from domyn_swarm.helpers.patterns import expand_brace_ranges


class PolarsBackend(DataBackend):
    name = "polars"

    def read(self, path: Path, *, limit: int | None = None, **kwargs) -> Any:
        import polars as pl

        patterns = expand_brace_ranges(str(path))
        parquet_input: str | list[str] = patterns[0] if len(patterns) == 1 else patterns

        use_scan = bool(kwargs.pop("use_scan", False) or kwargs.pop("lazy", False))
        kwargs.pop("streaming", None)
        if use_scan:
            lf = pl.scan_parquet(parquet_input, **kwargs)
            if limit:
                lf = lf.limit(limit)
            return lf

        df = pl.read_parquet(parquet_input, **kwargs)
        return df.head(limit) if limit else df

    def write(self, data: Any, path: Path, *, nshards: int | None = None, **kwargs) -> None:
        import polars as pl

        if not isinstance(data, pl.DataFrame):
            raise BackendError("Polars backend expects a polars.DataFrame for write().")

        path = Path(path)
        is_dir_target = path.is_dir()
        has_suffix = path.suffix != ""
        is_dir_output = is_dir_target or not has_suffix

        resolved_shards = max(1, int(nshards or 1))
        if not is_dir_output:
            # File output: behave like pandas backend and ignore sharding.
            path.parent.mkdir(parents=True, exist_ok=True)
            data.write_parquet(path, **kwargs)
            return

        path.mkdir(parents=True, exist_ok=True)
        total_rows = data.height
        width = max(1, len(str(resolved_shards - 1)))

        if total_rows == 0:
            data.write_parquet(path / f"data-{0:0{width}d}.parquet", **kwargs)
            return

        chunk_size = math.ceil(total_rows / resolved_shards)
        for shard_id in range(resolved_shards):
            start = shard_id * chunk_size
            if start >= total_rows:
                break
            data.slice(start, chunk_size).write_parquet(
                path / f"data-{shard_id:0{width}d}.parquet",
                **kwargs,
            )

    def schema(self, data: Any) -> dict[str, str]:
        return {k: str(v) for k, v in data.schema.items()}

    def to_pandas(self, data: Any) -> pd.DataFrame:
        import polars as pl

        if isinstance(data, pl.LazyFrame):
            data = data.collect()
        return data.to_pandas()

    def from_pandas(self, df: pd.DataFrame) -> Any:
        import polars as pl

        return pl.from_pandas(df)

    def to_arrow(self, data: Any) -> pa.Table:
        """Convert a polars DataFrame/LazyFrame to an Arrow table.

        Args:
            data: Polars DataFrame or LazyFrame.

        Returns:
            Arrow table containing the materialized data.
        """
        import polars as pl

        if isinstance(data, pl.LazyFrame):
            data = data.collect()
        return data.to_arrow()

    def from_arrow(self, table: pa.Table) -> Any:
        """Convert an Arrow table to a polars DataFrame."""
        import polars as pl

        return pl.from_arrow(table)

    def slice(self, data: Any, indices: list[int]) -> Any:
        return data.take(indices)

    def iter_batches(self, data: Any, *, batch_size: int) -> Iterable[Any]:
        """Yield batches from a polars DataFrame or LazyFrame.

        For DataFrames, this yields slices of the existing in-memory frame.
        For LazyFrames (e.g. returned by `scan_parquet`), this executes the query in streaming
        mode and yields materialized DataFrame chunks without collecting the full dataset up
        front.

        Args:
            data: Polars DataFrame or LazyFrame.
            batch_size: Maximum number of rows per yielded batch.

        Yields:
            Polars DataFrame batches.
        """
        import polars as pl

        if isinstance(data, pl.LazyFrame):
            # Note: collect_batches is currently marked unstable in polars.
            yield from data.collect_batches(chunk_size=batch_size, engine="streaming")
            return

        for start in range(0, data.height, batch_size):
            yield data.slice(start, batch_size)

    def iter_job_batches(
        self, data: Any, *, batch_size: int, id_col: str, input_col: str
    ) -> Iterable[JobBatch]:
        """Yield normalized batches for SwarmJob execution.

        Args:
            data: Polars DataFrame or LazyFrame.
            batch_size: Maximum number of rows per yielded batch.
            id_col: Column containing stable ids.
            input_col: Column containing job input values.

        Yields:
            `JobBatch` objects containing ids, items, and the batch DataFrame.
        """
        import polars as pl

        for batch in self.iter_batches(data, batch_size=batch_size):
            if not isinstance(batch, pl.DataFrame):
                raise BackendError("Polars iter_batches must yield polars.DataFrame for jobs.")
            yield JobBatch(
                ids=batch.get_column(id_col).to_list(),
                items=batch.get_column(input_col).to_list(),
                batch=batch,
            )
