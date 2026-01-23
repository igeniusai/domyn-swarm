import pytest

from domyn_swarm.jobs.api.runner import normalize_batch_outputs


@pytest.mark.parametrize(
    "expected_cols, outputs, rows, cols",
    [
        # single named column: scalar
        (["y"], [1, 2, 3], [1, 2, 3], ["y"]),
        # single named column: dict -> extract key
        (["score"], [{"score": 0.1}, {"score": 0.2}], [0.1, 0.2], ["score"]),
        # single named column: singleton tuple -> unwrap
        (["val"], [(10,), (20,)], [10, 20], ["val"]),
        # multi named columns: list/tuple positional
        (["a", "b"], [(1, 2), (3, 4)], [(1, 2), (3, 4)], ["a", "b"]),
        # multi named columns: dicts projected to order
        (
            ["a", "b"],
            [{"b": 2, "a": 1}, {"a": 3, "b": 4}],
            [[1, 2], [3, 4]],
            ["a", "b"],
        ),
        # no expected_cols: dict passthrough
        (
            None,
            [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
            [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
            None,
        ),
        # no expected_cols: scalar → synthesize 'output'
        (None, [42, 7], [42, 7], ["output"]),
    ],
)
def test_normalize_batch_outputs_happy(expected_cols, outputs, rows, cols):
    got_rows, got_cols = normalize_batch_outputs(outputs, expected_cols)
    assert got_rows == rows
    assert got_cols == cols


def test_normalize_batch_outputs_errors_tuple_without_names():
    with pytest.raises(ValueError):
        normalize_batch_outputs([(1, 2), (3, 4)], expected_cols=None)


def test_normalize_batch_outputs_errors_multi_cols_but_scalar():
    with pytest.raises(ValueError):
        normalize_batch_outputs([123, 456], expected_cols=["a", "b"])


def test_normalize_batch_outputs_errors_single_col_mismatch_tuple_len():
    with pytest.raises(ValueError):
        normalize_batch_outputs([(1, 2)], expected_cols=["only_one"])
