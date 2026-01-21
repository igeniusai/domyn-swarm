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

from pathlib import Path

import pandas as pd
import pytest

from domyn_swarm.data.backends.polars_backend import PolarsBackend


def test_polars_read_scan_parquet(tmp_path: Path) -> None:
    """Read parquet using scan_parquet and enforce limit behavior.

    Args:
        tmp_path: Pytest temporary directory.
    """
    pl = pytest.importorskip("polars")
    path = tmp_path / "input.parquet"
    pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).to_parquet(path)

    backend = PolarsBackend()
    out = backend.read(path, use_scan=True, limit=2)

    assert isinstance(out, pl.DataFrame)
    assert out.height == 2
