# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from domyn_swarm.checkpoint.manager import CheckpointManager


def test_checkpoint_manager_filters_completed(tmp_path):
    df = pd.DataFrame({"messages": ["a", "b", "c"], "result": [None, None, None]})
    ckpt_path = tmp_path / "checkpoint.parquet"

    done = df.loc[[0, 1]].copy()
    done["result"] = ["ra", "rb"]
    done.to_parquet(ckpt_path)

    mgr = CheckpointManager(
        ckpt_path,
        df,
        expected_output_cols="result",
        input_col="messages",
    )
    todo = mgr.filter_todo()

    assert list(todo.index) == [2]
    assert set(mgr.done_idx) == {0, 1}


def test_checkpoint_manager_missing_output_cols_raises(tmp_path):
    df = pd.DataFrame({"messages": ["a", "b"], "result": [None, None]})
    ckpt_path = tmp_path / "checkpoint.parquet"

    done = df.loc[[0]].copy()
    done.drop(columns=["result"]).to_parquet(ckpt_path)

    with pytest.raises(ValueError, match="missing expected output columns"):
        CheckpointManager(
            ckpt_path,
            df,
            expected_output_cols="result",
            input_col="messages",
        )


def test_checkpoint_manager_ignores_stray_rows(tmp_path):
    df = pd.DataFrame({"messages": ["a", "b", "c"], "result": [None, None, None]})
    ckpt_path = tmp_path / "checkpoint.parquet"

    done = pd.DataFrame({"messages": ["a", "x"], "result": ["ra", "rx"]}, index=[0, 99])
    done.to_parquet(ckpt_path)

    mgr = CheckpointManager(
        ckpt_path,
        df,
        expected_output_cols="result",
        input_col="messages",
    )
    assert set(mgr.done_idx) == {0}


def test_checkpoint_manager_fingerprint_mismatch(tmp_path):
    df = pd.DataFrame({"messages": ["a", "b"], "result": [None, None]})
    ckpt_path = tmp_path / "checkpoint.parquet"

    mgr = CheckpointManager(ckpt_path, df, expected_output_cols="result", input_col="messages")
    mgr.flush(["out_a"], [0], "result", df.index.to_numpy())

    df2 = pd.DataFrame({"messages": ["a", "c"], "result": [None, None]})
    with pytest.raises(ValueError, match="fingerprint"):
        CheckpointManager(ckpt_path, df2, expected_output_cols="result", input_col="messages")
