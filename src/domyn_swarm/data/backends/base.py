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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import pyarrow as pa


@dataclass(frozen=True)
class JobBatch:
    """Normalized batch representation for SwarmJobs.

    This shape is intended to be backend-agnostic: pandas, polars, and ray can all expose
    batches in a way that provides:
    - stable ids used for checkpointing/resume
    - the list of items passed into `SwarmJob.transform_items` / `transform_streaming`
    - the backend-native batch object, for optional downstream joins or debugging

    Args:
        ids: Stable identifiers for the rows in this batch.
        items: Items extracted from the input column for job execution.
        batch: Backend-native batch object (e.g., `pd.DataFrame`, `pl.DataFrame`, or a ray batch).
    """

    ids: list[Any]
    items: list[Any]
    batch: Any


class DataBackend(Protocol):
    name: str

    def read(self, path: Path, *, limit: int | None = None, **kwargs) -> Any: ...

    def write(self, data: Any, path: Path, *, nshards: int | None = None, **kwargs) -> None: ...

    def schema(self, data: Any) -> dict[str, str]: ...

    def to_pandas(self, data: Any) -> pd.DataFrame: ...

    def from_pandas(self, df: pd.DataFrame) -> Any: ...

    def to_arrow(self, data: Any) -> pa.Table:
        """Convert backend-native data into an Arrow table."""
        ...

    def from_arrow(self, table: pa.Table) -> Any:
        """Convert an Arrow table into backend-native data."""
        ...

    def slice(self, data: Any, indices: list[int]) -> Any: ...

    def iter_batches(self, data: Any, *, batch_size: int) -> Iterable[Any]: ...

    def iter_job_batches(
        self, data: Any, *, batch_size: int, id_col: str, input_col: str
    ) -> Iterable[JobBatch]:
        """Yield JobBatch objects for a given dataset.

        Args:
            data: Backend-native dataset object.
            batch_size: Maximum number of rows per yielded batch.
            id_col: Column name containing stable ids for checkpointing/resume.
            input_col: Column name containing input items for job execution.

        Yields:
            Normalized `JobBatch` instances.
        """
        ...


@dataclass(frozen=True)
class BackendError(RuntimeError):
    message: str

    def __str__(self) -> str:
        return self.message
