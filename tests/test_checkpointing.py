import os

import pandas as pd

from domyn_swarm.jobs.checkpointing import CheckpointManager


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
    manager.flush(out_list, new_ids=[0, 2], output_column_name="c", idx_map=idx_map)

    flushed_df = pd.read_parquet(path)
    assert set(flushed_df["c"]) == {"out0", "out2"}

    final = manager.finalize()
    assert isinstance(final, pd.DataFrame)
    assert os.path.exists(path) is False
