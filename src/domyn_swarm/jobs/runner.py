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

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from domyn_swarm.checkpoint.store import CheckpointStore, FlushBatch, ParquetShardStore
from domyn_swarm.helpers.logger import setup_logger
from domyn_swarm.jobs.base import OutputJoinMode, SwarmJob

logger = setup_logger(__name__)


def _normalize_batch_outputs(
    local_outputs: list[Any],
    expected_cols: list[str] | None,
) -> tuple[list[Any], list[str] | None]:
    """Normalize a batch of model outputs into (rows, columns-for-flush).

    Args:
        local_outputs: List of outputs from the model/job processing.
        expected_cols: Optional list of expected column names for output.

    Returns:
        tuple: A tuple containing:
            - rows: List of normalized output rows
            - columns: List of column names for flushing (None for dict path)

    Behavior:
        If expected_cols is given:
            - len==1:
                * scalar -> pass as-is
                * list/tuple len 1 -> unwrap
                * dict -> extract that key
            - len>1:
                * list/tuple -> positional align to expected_cols
                * dict -> project to expected_cols order
                * scalar -> error
        If expected_cols is None:
            - dict -> pass through with columns=None (store will use dict path)
            - scalar -> synthesize single column 'output'
            - list/tuple -> error (ambiguous without names)
    """
    if not local_outputs:
        return [], expected_cols

    first = local_outputs[0]

    # Case A: Caller provided explicit output column names
    if expected_cols:
        if len(expected_cols) == 1:
            if isinstance(first, dict):
                # convert dicts -> scalar per key
                key = expected_cols[0]
                rows = [o.get(key) for o in local_outputs]
                return rows, expected_cols
            elif isinstance(first, list | tuple):
                if len(first) != 1:
                    raise ValueError(
                        f"Expected single-column output {expected_cols}, "
                        f"but got list/tuple with length {len(first)}"
                    )
                rows = [o[0] for o in local_outputs]
                return rows, expected_cols
            else:
                # scalar per item
                return local_outputs, expected_cols
        else:
            # multi-column expected
            if isinstance(first, list | tuple):
                if any(len(o) != len(expected_cols) for o in local_outputs):
                    raise ValueError(f"Output rows length mismatch vs columns {expected_cols}")
                # keep as list/tuple; flush will index by position
                return local_outputs, expected_cols
            elif isinstance(first, dict):
                # convert dict -> row aligned to expected_cols
                rows = [[o.get(c) for c in expected_cols] for o in local_outputs]
                return rows, expected_cols
            else:
                raise ValueError(
                    f"Multiple output columns {expected_cols} require list/tuple or dict outputs"
                )

    # Case B: No expected columns provided
    #   - dict outputs → pass dicts with output_cols=None
    #   - scalar outputs → synthesize a single column 'output'
    #   - list/tuple outputs → ambiguous; require explicit output_cols
    if isinstance(first, dict):
        logger.debug("Output is dict; passing as-is.")
        return local_outputs, None  # dict path (flush with output_cols=None)
    elif isinstance(first, list | tuple):
        raise ValueError("List/tuple outputs require explicit `output_cols` naming.")
    else:
        logger.debug("Output is scalar; synthesizing 'output' column.")
        # scalar → single column auto-named 'output'
        return local_outputs, ["output"]


@dataclass
class RunnerConfig:
    id_col: str = "_row_id"
    checkpoint_every: int = 16
    total_concurrency: int | None = None  # reserved for future admission control


class JobRunner:
    """Unifies single- and multi-shard execution and decouples checkpointing.

    The job object is expected to expose a **streaming** transform:

        await job.transform_streaming(items: list[Any], *,
                                      on_flush: Callable[[list[int], list[Any]], Awaitable[None]],
                                      checkpoint_every: int) -> None

    Jobs are expected to implement `transform_streaming`, which is provided by
    `SwarmJob` and relies on `transform_items`.

    Parameters
    ----------
    store : CheckpointStore
        The persistence backend (local or cloud).
    cfg : RunnerConfig
        Runner tuning parameters.
    input_col : str
        Column name to read inputs from (e.g., "messages").
    output_cols : list[str] | None
        Output columns (None means the job returns dicts that will be normalized).
    """

    def __init__(self, store: CheckpointStore[pd.DataFrame], cfg: RunnerConfig | None = None):
        self.store = store
        self.cfg = cfg or RunnerConfig()

    async def run(
        self,
        job: SwarmJob,
        df: pd.DataFrame,
        *,
        input_col: str | list[str] | tuple[str],
        output_cols: list[str] | None,
        output_mode: OutputJoinMode | None = None,
    ) -> pd.DataFrame:
        """
        Run a streaming job and either:
          - append outputs to original df (APPEND), or
          - return only id + outputs (REPLACE).

        output_cols:
          - None -> expect dict outputs per item (or scalar => auto 'output')
          - [name] -> scalar outputs per item
          - [c1, c2, ...] -> list/tuple (positional) or dict outputs (named)
        """
        if input_col not in df.columns:
            raise ValueError(f"input_col '{input_col}' not found in DataFrame")
        mode = output_mode or getattr(job, "output_mode", OutputJoinMode.APPEND)
        if isinstance(mode, str):
            mode = OutputJoinMode(mode)

        df = df.copy(deep=False)
        if self.cfg.id_col not in df.columns:
            df[self.cfg.id_col] = df.index

        logger.debug("Length of input DataFrame: %d", len(df))
        todo = self.store.prepare(df, self.cfg.id_col)
        logger.debug("Number of items to process after checkpointing: %d", len(todo))
        ids: list[Any] = todo[self.cfg.id_col].tolist()
        items: list[Any] = todo[input_col].tolist()

        # Normalize a batch of outputs to what flush expects
        # Returns (rows, cols_for_flush) where cols_for_flush can be None => dict path
        async def _on_flush(local_indices: list[int], local_outputs: list[Any]):
            batch_ids = [ids[i] for i in local_indices]
            rows, cols = _normalize_batch_outputs(local_outputs, output_cols)
            await self.store.flush(
                FlushBatch(ids=batch_ids, rows=rows),
                output_cols=cols,  # None => dict path; else aligned columns
            )

        if not hasattr(job, "transform_streaming"):
            raise TypeError(
                "Job must implement "
                "`transform_streaming(items, on_flush=..., checkpoint_every=...)`"
            )

        await job.transform_streaming(
            items,
            on_flush=_on_flush,
            checkpoint_every=self.cfg.checkpoint_every,
        )
        out_df = await asyncio.to_thread(self.store.finalize)
        if out_df.index.name == self.cfg.id_col:
            if self.cfg.id_col in out_df.columns:
                # id is already a column → keep it and drop the index
                out_df = out_df.reset_index(drop=True)
            else:
                # id only lives in the index → move it to a column
                out_df = out_df.reset_index()

        if mode == OutputJoinMode.APPEND:
            # left-join to preserve original row order and columns
            return df.merge(out_df, on=self.cfg.id_col, how="left")
        elif mode == OutputJoinMode.IO_ONLY:
            # keep only id + inputs + outputs
            out_df = df.merge(out_df, on=self.cfg.id_col, how="left")
            if output_cols:
                keep = [self.cfg.id_col, input_col, *output_cols]
            else:
                output_columns = [
                    c for c in out_df.columns if c not in (self.cfg.id_col, input_col)
                ]
                keep = [self.cfg.id_col, input_col, *output_columns]
        else:
            if output_cols:
                keep = [self.cfg.id_col, *output_cols]
            else:
                output_columns = [c for c in out_df.columns if c != self.cfg.id_col]
                keep = [self.cfg.id_col, *output_columns]
        return out_df.loc[:, keep]


async def run_sharded(
    job_factory: Callable[[], Any],
    df: pd.DataFrame,
    *,
    input_col: str,
    output_cols: list[str] | None,
    store_uri: str,  # e.g. file:///..., s3://...
    nshards: int = 1,
    cfg: RunnerConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or RunnerConfig()
    if nshards <= 1:
        store = ParquetShardStore(store_uri)
        return await JobRunner(store, cfg).run(
            job_factory(), df, input_col=input_col, output_cols=output_cols
        )

    # split by index and process concurrently
    indices = np.array_split(df.index, nshards)
    import asyncio

    async def _one_shard(shard_id: int, idx):
        sub = df.loc[idx].copy(deep=False)
        su = store_uri.replace(".parquet", f"_shard{shard_id}.parquet")
        store = ParquetShardStore(su)
        runner = JobRunner(store, cfg)
        return await runner.run(job_factory(), sub, input_col=input_col, output_cols=output_cols)

    parts = await asyncio.gather(*[_one_shard(i, idx) for i, idx in enumerate(indices)])
    return pd.concat(parts).sort_index()
