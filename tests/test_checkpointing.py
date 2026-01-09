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

from domyn_swarm.checkpoint.manager import CheckpointManager


def test_checkpoint_manager_filters_and_flushes(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_path / "checkpoint.parquet"
    manager = CheckpointManager(path, df)

    todo = manager.filter_todo()
    assert len(todo) == 3

    # Simulate batched output: pretend todo_df has indices 0,1,2 → original index map
    idx_map = [0, 1, 2]

    # Simulate out_list aligned with todo_df, e.g., results of inference
    out_list = ["out0", "out1", "out2"]

    # Simulate flushing rows 0 and 2 (in todo_df's index space)
    manager.flush(out_list, new_ids=[0, 2], output_cols="c", idx_map=idx_map)
    manager.finalize()

    flushed_df = pd.read_parquet(path)
    assert set(flushed_df["c"]) == {"out0", "out2"}

    final = manager.finalize()
    assert isinstance(final, pd.DataFrame)
    # Disabled as long as checkpoint deletion is not implemented
    # assert os.path.exists(path) is False
