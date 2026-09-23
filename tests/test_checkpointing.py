# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import pandas as pd

from domyn_swarm.checkpoint.manager import CheckpointManager


def test_checkpoint_manager_filters_and_flushes(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_path / "checkpoint.parquet"
    manager = CheckpointManager(path, df)

    todo = manager.filter_todo()
    assert len(todo) == 3

    index_map = [0, 1, 2]
    out_list = ["out0", "out1", "out2"]

    manager.flush(out_list, new_ids=[0, 2], output_cols="c", idx_map=index_map)
    manager.finalize()

    flushed_df = pd.read_parquet(path)
    assert set(flushed_df["c"]) == {"out0", "out2"}

    final = manager.finalize()
    assert isinstance(final, pd.DataFrame)
