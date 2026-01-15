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

from domyn_swarm.data.backends.base import BackendError, DataBackend


class PolarsBackend(DataBackend):
    name = "polars"

    def read(self, path: Path, *, limit: int | None = None, **kwargs) -> Any:
        import polars as pl

        df = pl.read_parquet(path, **kwargs)
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
        return data.to_pandas()

    def from_pandas(self, df: pd.DataFrame) -> Any:
        import polars as pl

        return pl.from_pandas(df)

    def slice(self, data: Any, indices: list[int]) -> Any:
        return data.take(indices)

    def iter_batches(self, data: Any, *, batch_size: int) -> Iterable[Any]:
        for start in range(0, data.height, batch_size):
            yield data.slice(start, batch_size)
