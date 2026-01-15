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
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa

from domyn_swarm.data.backends.base import BackendError, DataBackend


class RayBackend(DataBackend):
    name = "ray"

    def read(self, path: Path, *, limit: int | None = None, **kwargs) -> Any:
        import ray.data as rd

        ds = rd.read_parquet(str(path), **kwargs)
        return ds.limit(limit) if limit else ds

    def write(self, data: Any, path: Path, *, nshards: int | None = None, **kwargs) -> None:
        # Ray write_parquet writes a dataset (directory). We currently don't provide
        # deterministic shard control via `nshards`; treat `None`/`1` as default behavior.
        if nshards not in (None, 1):
            raise BackendError("Ray backend does not support controlling shard writes yet.")
        data.write_parquet(str(path), **kwargs)

    def schema(self, data: Any) -> dict[str, str]:
        return {field.name: str(field.type) for field in data.schema().types}

    def to_pandas(self, data: Any) -> pd.DataFrame:
        return data.to_pandas()

    def from_pandas(self, df: pd.DataFrame) -> Any:
        import ray.data as rd

        return rd.from_pandas(df)

    def to_arrow(self, data: Any) -> pa.Table:
        """Ray backend does not support Arrow conversion."""
        raise BackendError("Arrow conversion is not supported for the ray backend.")

    def from_arrow(self, table: pa.Table) -> Any:
        """Ray backend does not support Arrow conversion."""
        raise BackendError("Arrow conversion is not supported for the ray backend.")

    def slice(self, data: Any, indices: list[int]) -> Any:
        # Fallback: slice via pandas conversion (native slicing can be added later).
        df = data.to_pandas()
        return df.iloc[indices]

    def iter_batches(self, data: Any, *, batch_size: int) -> Iterable[Any]:
        return data.iter_batches(batch_size=batch_size)
