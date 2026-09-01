# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the FrameOps adapters."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from domyn_swarm.data.backends.registry import get_backend
from domyn_swarm.jobs.execution.frame_ops import ArrowFrameOps, PandasFrameOps


@pytest.fixture
def pandas_ops() -> PandasFrameOps:
    return PandasFrameOps(get_backend("pandas"))


@pytest.fixture
def arrow_ops() -> ArrowFrameOps:
    return ArrowFrameOps(get_backend("pandas"))


def test_pandas_ensure_id_uses_the_index(pandas_ops: PandasFrameOps) -> None:
    """The pandas adapter derives missing ids from the frame index."""
    df = pd.DataFrame({"messages": [1, 2]}, index=[7, 8])

    out = pandas_ops.ensure_id(df, "_row_id")

    assert out["_row_id"].tolist() == [7, 8]


def test_arrow_ensure_id_renames_a_known_index_column(arrow_ops: ArrowFrameOps) -> None:
    """The Arrow adapter recovers ids from a pandas-written index column.

    This candidate-rename behaviour is specific to Arrow and must not be
    unified with the pandas adapter.
    """
    table = pa.table({"__index_level_0__": [5, 6], "messages": [1, 2]})

    out = arrow_ops.ensure_id(table, "_row_id")

    assert out.column("_row_id").to_pylist() == [5, 6]


def test_arrow_ensure_id_falls_back_to_a_range(arrow_ops: ArrowFrameOps) -> None:
    """With no candidate column, Arrow ids are positional."""
    table = pa.table({"messages": [1, 2, 3]})

    out = arrow_ops.ensure_id(table, "_row_id")

    assert out.column("_row_id").to_pylist() == [0, 1, 2]


def test_pandas_concat_sorts_by_index(pandas_ops: PandasFrameOps) -> None:
    """Row order is restored after shards complete out of order."""
    a = pd.DataFrame({"v": ["b"]}, index=[1])
    b = pd.DataFrame({"v": ["a"]}, index=[0])

    out = pandas_ops.concat([a, b])

    assert out["v"].tolist() == ["a", "b"]


def test_arrow_concat_preserves_part_order(arrow_ops: ArrowFrameOps) -> None:
    """Arrow concatenation is positional, with no re-sort."""
    a = pa.table({"v": ["b"]})
    b = pa.table({"v": ["a"]})

    out = arrow_ops.concat([a, b])

    assert out.column("v").to_pylist() == ["b", "a"]


@pytest.mark.parametrize("ops_name", ["pandas_ops", "arrow_ops"])
def test_filter_out_ids_removes_done_rows(ops_name, request) -> None:
    """Both adapters drop rows whose id is already done."""
    ops = request.getfixturevalue(ops_name)
    frame = ops.coerce(pd.DataFrame({"doc_id": [1, 2, 3], "messages": [1, 2, 3]}), "doc_id")

    out = ops.filter_out_ids(frame, "doc_id", {2})

    assert sorted(_ids(out, "doc_id")) == [1, 3]


def _ids(frame, id_col: str) -> list:
    """Read an id column from either frame type."""
    if isinstance(frame, pa.Table):
        return frame.column(id_col).to_pylist()
    return frame[id_col].tolist()


@pytest.mark.parametrize("ops_name", ["pandas_ops", "arrow_ops"])
def test_shard_indices_partition_every_row_exactly_once(ops_name, request) -> None:
    """Sharding is a partition: no row lost, none duplicated."""
    ops = request.getfixturevalue(ops_name)
    frame = ops.coerce(
        pd.DataFrame({"doc_id": list(range(10)), "messages": list(range(10))}), "doc_id"
    )

    shards = ops.shard_indices(frame, "doc_id", "id", 3)

    seen = sorted(int(i) for idx in shards for i in idx)
    assert seen == list(range(10))


def test_arrow_coerce_preseeds_id_from_a_non_default_pandas_index(
    arrow_ops: ArrowFrameOps,
) -> None:
    """A pandas frame's original index survives coerce + ensure_id as ids.

    `ArrowFrameOps.coerce` must preseed `id_col` from the pandas index before
    handing the frame to backend conversion, because Arrow conversion itself
    drops the index. Without the preseed, `ensure_id` would fall through to
    its positional-range fallback and silently renumber every row.
    """
    df = pd.DataFrame({"messages": [1, 2, 3]}, index=[10, 11, 12])

    table = arrow_ops.coerce(df, "doc_id")
    out = arrow_ops.ensure_id(table, "doc_id")

    assert out.column("doc_id").to_pylist() == [10, 11, 12]


def test_pandas_take_selects_by_label(pandas_ops: PandasFrameOps) -> None:
    """`take` selects rows by label, matching `shard_mode="index"` semantics."""
    df = pd.DataFrame({"v": ["a", "b", "c"]}, index=[10, 20, 30])

    out = pandas_ops.take(df, np.array([20, 10]))

    assert out["v"].tolist() == ["b", "a"]


def test_pandas_take_positional_selects_by_position(pandas_ops: PandasFrameOps) -> None:
    """`take_positional` selects rows by position, ignoring the index labels."""
    df = pd.DataFrame({"v": ["a", "b", "c"]}, index=[10, 20, 30])

    out = pandas_ops.take_positional(df, np.array([1, 0]))

    assert out["v"].tolist() == ["b", "a"]


def test_arrow_take_and_take_positional_both_select_by_position(
    arrow_ops: ArrowFrameOps,
) -> None:
    """Arrow has no separate row labels: `take` and `take_positional` agree."""
    table = pa.table({"v": ["a", "b", "c"]})

    by_take = arrow_ops.take(table, np.array([1, 0]))
    by_take_positional = arrow_ops.take_positional(table, np.array([1, 0]))

    assert by_take.column("v").to_pylist() == ["b", "a"]
    assert by_take_positional.column("v").to_pylist() == ["b", "a"]
