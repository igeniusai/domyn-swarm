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

from domyn_swarm.checkpoint.store import ParquetShardStore


def test_finalize_reads_legacy_index_column_as_id(tmp_path):
    base = tmp_path / "legacy.parquet"

    # Simulate a legacy parquet that has the id in the parquet index column
    # (pandas often writes this as '__index_level_0__' when the index is unnamed).
    df = pd.DataFrame({"result": ["a", "b"]}, index=[0, 1])
    df.to_parquet(base)

    store = ParquetShardStore(f"file://{base}")
    store.id_col = "_row_id"
    out = store.finalize()

    assert out.index.name == "_row_id"
    assert out.loc[0, "result"] == "a"
    assert out.loc[1, "result"] == "b"
