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

import pandas as pd
import pytest

from domyn_swarm.data.backends.base import JobBatch
from domyn_swarm.data.backends.pandas_backend import PandasBackend
from domyn_swarm.data.backends.polars_backend import PolarsBackend


def test_pandas_iter_job_batches() -> None:
    """Yield normalized JobBatch objects from pandas data.

    This validates the common shape expected by runners, regardless of the underlying backend.
    """
    df = pd.DataFrame({"_row_id": [0, 1, 2], "messages": ["a", "b", "c"]})
    backend = PandasBackend()

    batches = list(
        backend.iter_job_batches(df, batch_size=2, id_col="_row_id", input_col="messages")
    )
    assert [type(b) for b in batches] == [JobBatch, JobBatch]
    assert batches[0].ids == [0, 1]
    assert batches[0].items == ["a", "b"]
    assert batches[1].ids == [2]
    assert batches[1].items == ["c"]


def test_polars_iter_job_batches() -> None:
    """Yield normalized JobBatch objects from polars data (DataFrame and LazyFrame)."""
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"_row_id": [0, 1, 2], "messages": ["a", "b", "c"]})
    backend = PolarsBackend()

    batches = list(
        backend.iter_job_batches(df, batch_size=2, id_col="_row_id", input_col="messages")
    )
    assert [type(b) for b in batches] == [JobBatch, JobBatch]
    assert batches[0].ids == [0, 1]
    assert batches[0].items == ["a", "b"]

    lf = df.lazy()
    lazy_batches = list(
        backend.iter_job_batches(lf, batch_size=2, id_col="_row_id", input_col="messages")
    )
    assert lazy_batches[0].ids == [0, 1]
    assert lazy_batches[0].items == ["a", "b"]
