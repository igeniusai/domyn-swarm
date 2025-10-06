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

from domyn_swarm.helpers.io import (
    is_folder,
    load_dataframe,
    path_exists,
    save_dataframe,
    to_path,
)
from domyn_swarm.utils.env_path import EnvPath


@pytest.fixture
def sample_df():
    return pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})


def test_load_and_save_parquet(tmp_path, sample_df):
    file_path = tmp_path / "data.parquet"
    save_dataframe(sample_df, file_path)
    loaded = load_dataframe(file_path)
    pd.testing.assert_frame_equal(loaded, sample_df)


def test_load_and_save_csv(tmp_path, sample_df):
    file_path = tmp_path / "data.csv"
    save_dataframe(sample_df, file_path)
    loaded = load_dataframe(file_path)
    pd.testing.assert_frame_equal(loaded, sample_df)


def test_load_and_save_jsonl(tmp_path, sample_df):
    file_path = tmp_path / "data.jsonl"
    save_dataframe(sample_df, file_path)
    loaded = load_dataframe(file_path)
    pd.testing.assert_frame_equal(loaded, sample_df)


def test_load_dataframe_unsupported(tmp_path):
    with pytest.raises(ValueError):
        load_dataframe(tmp_path / "data.txt")


def test_save_dataframe_unsupported(tmp_path, sample_df):
    with pytest.raises(ValueError):
        save_dataframe(sample_df, tmp_path / "data.unsupported")


def test_to_path_str():
    path_str = "/tmp/some_path"
    assert isinstance(to_path(path_str), EnvPath)


def test_to_path_envpath():
    p = EnvPath("/tmp/some_path")
    assert to_path(p) is p


def test_is_folder(tmp_path):
    assert is_folder(str(tmp_path)) is True


def test_path_exists(tmp_path):
    test_file = tmp_path / "file.txt"
    test_file.write_text("hello")
    assert path_exists(str(test_file))
