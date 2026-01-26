import pandas as pd
import pytest

from domyn_swarm.jobs.api.runner import (
    JobRunner,
    OutputJoinMode,  # your enum
    RunnerConfig,
)


class FakeStore:
    """Minimal CheckpointStore behavior for run() tests."""

    def __init__(self, uri: str = "mem://test"):
        self.uri = uri
        self.id_col = None
        self._partials = []  # list of DataFrames to concat

    def prepare(self, df: pd.DataFrame, id_col: str) -> pd.DataFrame:
        self.id_col = id_col
        return df

    async def flush(self, batch, output_cols):
        # build a small DataFrame chunk from the FlushBatch + columns
        ids = list(batch.ids)
        if output_cols is None:
            # dict path
            chunk = pd.DataFrame(batch.rows)
            chunk[self.id_col] = ids
        else:
            if len(output_cols) == 1:
                chunk = pd.DataFrame({self.id_col: ids, output_cols[0]: batch.rows})
            else:
                # rows are either list/tuple aligned to output_cols
                chunk = pd.DataFrame(batch.rows, columns=output_cols)
                chunk[self.id_col] = ids
        # Keep id first for easier merges later
        cols = [self.id_col] + [c for c in chunk.columns if c != self.id_col]
        self._partials.append(chunk.loc[:, cols])

    def finalize(self) -> pd.DataFrame:
        if not self._partials:
            return pd.DataFrame(columns=[self.id_col])
        df = pd.concat(self._partials, ignore_index=True)
        # emulate store behavior: id as index sometimes
        df = df.set_index(self.id_col, drop=False)
        return df


class FakeJobScalarSingleCol:
    """transform_streaming returns a single scalar per item (e.g., score)."""

    output_mode = OutputJoinMode.APPEND
    max_concurrency = 8
    retries = 0

    async def transform_streaming(self, items, *, on_flush, checkpoint_every):
        # one flush containing all rows
        outputs = [f"{x}-out" for x in items]
        await on_flush(list(range(len(items))), outputs)


class FakeJobDictNoCols:
    """transform_streaming returns dicts per item and caller does not pass output_cols."""

    output_mode = OutputJoinMode.IO_ONLY
    max_concurrency = 8
    retries = 0

    async def transform_streaming(self, items, *, on_flush, checkpoint_every):
        outputs = [{"a": len(x), "b": x.upper()} for x in items]
        await on_flush(list(range(len(items))), outputs)


class FakeJobMultiColsTuple:
    """transform_streaming returns tuples per item; caller passes output_cols=['a','b']"""

    output_mode = OutputJoinMode.REPLACE
    max_concurrency = 8
    retries = 0

    async def transform_streaming(self, items, *, on_flush, checkpoint_every):
        outputs = [(len(x), x[::-1]) for x in items]
        await on_flush(list(range(len(items))), outputs)


@pytest.mark.asyncio
async def test_jobrunner_run_append_with_scalar_single_col():
    df = pd.DataFrame({"messages": [f"m{i}" for i in range(4)]})
    store = FakeStore()
    runner = JobRunner(store, RunnerConfig())  # default id_col expected by your code

    job = FakeJobScalarSingleCol()
    out = await runner.run(
        job,
        df,
        input_col="messages",
        output_cols=["result"],
        output_mode=OutputJoinMode.APPEND,
    )

    # Should preserve original columns + 'result'
    expected = {"messages", runner.cfg.id_col, "result"}
    assert expected <= set(out.columns)
    # Left join semantics → order preserved
    assert out["result"].tolist() == [f"m{i}-out" for i in range(4)]


@pytest.mark.asyncio
async def test_jobrunner_run_io_only_with_dict_outputs_no_explicit_cols():
    df = pd.DataFrame({"messages": ["hi", "there", "you"]})
    store = FakeStore()
    runner = JobRunner(store, RunnerConfig())

    job = FakeJobDictNoCols()
    out = await runner.run(
        job,
        df,
        input_col="messages",
        output_cols=None,
        output_mode=OutputJoinMode.IO_ONLY,
    )

    # Should keep only id + input + materialized dict keys
    id_col = runner.cfg.id_col
    assert set(out.columns) == {id_col, "messages", "a", "b"}
    # sanity on values
    assert out["a"].tolist() == [2, 5, 3]
    assert out["b"].tolist() == ["HI", "THERE", "YOU"]


@pytest.mark.asyncio
async def test_jobrunner_run_replace_with_multi_cols_tuple():
    df = pd.DataFrame({"messages": ["abc", "xy"]})
    store = FakeStore()
    runner = JobRunner(store, RunnerConfig())

    job = FakeJobMultiColsTuple()
    out = await runner.run(
        job,
        df,
        input_col="messages",
        output_cols=["alen", "rev"],
        output_mode=OutputJoinMode.REPLACE,
    )

    id_col = runner.cfg.id_col
    # Only id + outputs
    assert set(out.columns) == {id_col, "alen", "rev"}
    assert out["alen"].tolist() == [3, 2]
    assert out["rev"].tolist() == ["cba", "yx"]


@pytest.mark.asyncio
async def test_jobrunner_run_raises_on_tuple_outputs_without_names():
    df = pd.DataFrame({"messages": ["a", "bb"]})
    store = FakeStore()
    runner = JobRunner(store, RunnerConfig())

    class BadJob:
        output_mode = OutputJoinMode.APPEND
        max_concurrency = 8
        retries = 0

        async def transform_streaming(self, items, *, on_flush, checkpoint_every):
            outputs = [(1, 2), (3, 4)]  # tuple outputs but no output_cols passed
            await on_flush(list(range(len(items))), outputs)

    with pytest.raises(ValueError):
        await runner.run(
            BadJob(),
            df,
            input_col="messages",
            output_cols=None,
            output_mode=OutputJoinMode.APPEND,
        )
