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

from dataclasses import dataclass
from typing import Any

import pandas as pd
import pyarrow as pa

from domyn_swarm.data.backends.base import JobBatch
from domyn_swarm.data.backends.ray_backend import RayBackend


@dataclass(frozen=True)
class _FakeSchema:
    """Minimal schema-like object for testing `RayBackend.schema()`."""

    names: list[str]
    types: list[Any]


class _FakeRayDataset:
    """Minimal Dataset-like object for testing Ray backend helpers."""

    def __init__(self, *, schema_obj: Any, batches: list[Any]):
        self._schema_obj = schema_obj
        self._batches = batches

    def schema(self) -> Any:
        return self._schema_obj

    def iter_batches(self, *, batch_size: int) -> Any:
        yield from self._batches


def test_ray_backend_schema_names_and_types() -> None:
    """Extract schema from a ray-like schema object exposing names/types."""
    backend = RayBackend()
    ds = _FakeRayDataset(
        schema_obj=_FakeSchema(names=["a", "b"], types=[pa.int64(), pa.string()]),
        batches=[],
    )
    assert backend.schema(ds) == {"a": "int64", "b": "string"}


def test_ray_backend_schema_pyarrow_schema() -> None:
    """Extract schema from an Arrow schema fallback path."""
    backend = RayBackend()
    ds = _FakeRayDataset(
        schema_obj=pa.schema([pa.field("a", pa.int64()), pa.field("b", pa.string())]),
        batches=[],
    )
    assert backend.schema(ds) == {"a": "int64", "b": "string"}


def test_ray_backend_iter_job_batches_handles_common_batch_shapes() -> None:
    """Yield JobBatch from pandas/arrow/dict batch shapes without ray installed."""
    backend = RayBackend()
    df_batch = pd.DataFrame({"_row_id": [0, 1], "messages": ["a", "b"]})
    arrow_batch = pa.Table.from_pydict({"_row_id": [2], "messages": ["c"]})
    dict_batch = {"_row_id": [3], "messages": ["d"]}

    ds = _FakeRayDataset(
        schema_obj=_FakeSchema(names=["_row_id", "messages"], types=[pa.int64(), pa.string()]),
        batches=[df_batch, arrow_batch, dict_batch],
    )

    out = list(backend.iter_job_batches(ds, batch_size=2, id_col="_row_id", input_col="messages"))
    assert [type(b) for b in out] == [JobBatch, JobBatch, JobBatch]
    assert out[0].ids == [0, 1]
    assert out[0].items == ["a", "b"]
    assert out[1].ids == [2]
    assert out[1].items == ["c"]
    assert out[2].ids == [3]
    assert out[2].items == ["d"]
