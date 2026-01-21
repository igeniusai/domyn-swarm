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

from domyn_swarm.data.backends.base import DataBackend
from domyn_swarm.helpers.io import load_dataframe, save_dataframe


class PandasBackend(DataBackend):
    name = "pandas"

    def read(self, path: Path, *, limit: int | None = None, **kwargs) -> pd.DataFrame:
        return load_dataframe(path, limit=limit, read_kwargs=kwargs or None)

    def write(
        self, data: pd.DataFrame, path: Path, *, nshards: int | None = None, **kwargs
    ) -> None:
        save_dataframe(data, path, nshards=nshards, write_kwargs=kwargs or None)

    def schema(self, data: pd.DataFrame) -> dict[str, str]:
        return {str(k): str(v) for k, v in data.dtypes.items()}

    def to_pandas(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    def from_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def to_arrow(self, data: pd.DataFrame) -> pa.Table:
        """Convert a pandas DataFrame to an Arrow table."""
        return pa.Table.from_pandas(data, preserve_index=False)

    def from_arrow(self, table: pa.Table) -> pd.DataFrame:
        """Convert an Arrow table to a pandas DataFrame."""
        return table.to_pandas(ignore_metadata=True)

    def slice(self, data: pd.DataFrame, indices: list[int]) -> pd.DataFrame:
        return data.iloc[indices]

    def iter_batches(self, data: pd.DataFrame, *, batch_size: int) -> Iterable[pd.DataFrame]:
        for start in range(0, len(data), batch_size):
            yield data.iloc[start : start + batch_size]

    def iter_job_batches(
        self, data: pd.DataFrame, *, batch_size: int, id_col: str, input_col: str
    ) -> Iterable[Any]:
        """Yield normalized batches for SwarmJob execution.

        Args:
            data: Input pandas DataFrame.
            batch_size: Maximum number of rows per yielded batch.
            id_col: Column containing stable ids.
            input_col: Column containing job input values.

        Yields:
            `JobBatch` objects containing ids, items, and the batch DataFrame.
        """
        from domyn_swarm.data.backends.base import JobBatch

        for batch in self.iter_batches(data, batch_size=batch_size):
            yield JobBatch(
                ids=batch[id_col].to_list(),
                items=batch[input_col].to_list(),
                batch=batch,
            )
