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

from domyn_swarm.checkpoint.manager import CheckpointManager


def test_checkpoint_manager_filters_completed(tmp_path):
    df = pd.DataFrame({"messages": ["a", "b", "c"], "result": [None, None, None]})
    ckpt_path = tmp_path / "checkpoint.parquet"

    done = df.loc[[0, 1]].copy()
    done["result"] = ["ra", "rb"]
    done.to_parquet(ckpt_path)

    mgr = CheckpointManager(ckpt_path, df, expected_output_cols="result")
    todo = mgr.filter_todo()

    assert list(todo.index) == [2]
    assert set(mgr.done_idx) == {0, 1}


def test_checkpoint_manager_missing_output_cols_raises(tmp_path):
    df = pd.DataFrame({"messages": ["a", "b"], "result": [None, None]})
    ckpt_path = tmp_path / "checkpoint.parquet"

    done = df.loc[[0]].copy()
    done.drop(columns=["result"]).to_parquet(ckpt_path)

    with pytest.raises(ValueError, match="missing expected output columns"):
        CheckpointManager(ckpt_path, df, expected_output_cols="result")


def test_checkpoint_manager_ignores_stray_rows(tmp_path):
    df = pd.DataFrame({"messages": ["a", "b", "c"], "result": [None, None, None]})
    ckpt_path = tmp_path / "checkpoint.parquet"

    done = pd.DataFrame({"messages": ["a", "x"], "result": ["ra", "rx"]}, index=[0, 99])
    done.to_parquet(ckpt_path)

    mgr = CheckpointManager(ckpt_path, df, expected_output_cols="result")
    assert set(mgr.done_idx) == {0}
