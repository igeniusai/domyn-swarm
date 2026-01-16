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
import inspect
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from domyn_swarm.checkpoint.store import InMemoryStore, ParquetShardStore
from domyn_swarm.data import BackendError, get_backend
from domyn_swarm.data.backends.base import DataBackend
from domyn_swarm.jobs.arrow_runner import run_arrow_job
from domyn_swarm.jobs.base import SwarmJob
from domyn_swarm.jobs.runner import JobRunner, RunnerConfig


def _ensure_arrow_id(table: pa.Table, id_col: str) -> pa.Table:
    """Ensure an Arrow table contains the id column.

    Args:
        table: Input Arrow table.
        id_col: Column name for row ids.

    Returns:
        Arrow table with the id column present.
    """
    if id_col in table.column_names:
        return table
    ids = pa.array(range(len(table)))
    return table.append_column(id_col, ids)


async def _run_non_ray_arrow(
    *,
    job_factory: Callable[[], Any],
    backend: Any,
    data: Any,
    input_col: str,
    output_cols: list[str] | None,
    nshards: int,
    store_uri: str | None,
    checkpoint_every: int,
    checkpointing: bool,
) -> pa.Table:
    """Run the Arrow runner path for non-ray backends.

    Args:
        job_factory: Callable producing a SwarmJob instance.
        backend: Data backend used for Arrow conversion.
        data: Backend-native data or Arrow table.
        input_col: Input column name in the dataset.
        output_cols: Output columns to produce (None for dict outputs).
        nshards: Number of shards to split the input into.
        store_uri: Base checkpoint store URI.
        checkpoint_every: Flush interval in items.
        checkpointing: Whether checkpointing is enabled.

    Returns:
        Arrow table containing job outputs (and inputs depending on output mode).
    """
    table = data if isinstance(data, pa.Table) else backend.to_arrow(data)
    if checkpointing and store_uri is None:
        raise ValueError("store_uri is required when checkpointing is enabled.")
    if nshards <= 1:
        return await run_arrow_job(
            job_factory,
            table,
            input_col=input_col,
            output_cols=output_cols,
            store_uri=store_uri,
            checkpoint_every=checkpoint_every,
            checkpointing=checkpointing,
        )
    if not checkpointing:
        raise ValueError("Sharded execution requires checkpointing to be enabled.")

    table = _ensure_arrow_id(table, "_row_id")
    indices = np.array_split(np.arange(table.num_rows), nshards)

    async def _one(i, idx):
        assert store_uri is not None
        sub = table.take(pa.array(idx, type=pa.int64()))
        su = store_uri.replace(".parquet", f"_shard{i}.parquet")
        return await run_arrow_job(
            job_factory,
            sub,
            input_col=input_col,
            output_cols=output_cols,
            store_uri=su,
            checkpoint_every=checkpoint_every,
            checkpointing=checkpointing,
        )

    parts = await asyncio.gather(*[_one(i, idx) for i, idx in enumerate(indices)])
    return pa.concat_tables(parts)


def _has_new_api(job: SwarmJob) -> bool:
    return resolve_job_api(job) == "new"


def _is_overridden(obj: object, method: str, base: type) -> bool:
    """
    True iff `method` is implemented by `type(obj)` (or an intermediate base),
    not by `base` itself.
    """
    cls = type(obj)
    for c in cls.mro():
        if method in c.__dict__:
            return c is not base  # first definition found is not the base → overridden
    return False  # method not found at all (shouldn't happen here)


def resolve_job_api(job: SwarmJob) -> str:
    if getattr(type(job), "api_version", 1) >= 2:
        return "new"
    # fallback to override-based inference for older subclasses
    if _is_overridden(job, "transform_items", SwarmJob) or _is_overridden(
        job, "transform_streaming", SwarmJob
    ):
        return "new"
    if _is_overridden(job, "transform", SwarmJob) or _is_overridden(job, "run", SwarmJob):
        return "old"
    return "old"


async def run_job_unified(
    job_factory: Callable[[], Any],
    data: Any,
    *,
    input_col: str,
    output_cols: list[str] | None,
    nshards: int = 1,
    store_uri: str | None = None,
    checkpoint_every: int = 16,
    tag: str | None = None,
    checkpoint_dir: str | None = None,
    data_backend: str | None = None,
    native_backend: bool | None = None,
    checkpointing: bool = True,
    runner: str = "pandas",
) -> Any:
    job_probe = job_factory()

    if not _has_new_api(job_probe):
        raise TypeError(
            "Job must implement the streaming API (transform_items/transform_streaming)."
        )

    backend_name = data_backend or getattr(job_probe, "data_backend", None) or "pandas"
    try:
        backend: DataBackend = get_backend(backend_name)
    except BackendError as exc:
        raise RuntimeError(str(exc)) from exc

    if backend.name == "ray":
        native = bool(native_backend) if native_backend is not None else True
        if not native:
            raise ValueError("Ray backend requires native execution (native_backend=True).")
        return await _run_job_ray(
            job_factory,
            data,
            input_col=input_col,
            output_cols=output_cols or job_probe.default_output_cols,
            batch_size=getattr(job_probe, "native_batch_size", None) or checkpoint_every,
            output_mode=job_probe.output_mode,
        )

    # Non-ray:
    if runner == "arrow":
        return await _run_non_ray_arrow(
            job_factory=job_factory,
            backend=backend,
            data=data,
            input_col=input_col,
            output_cols=output_cols or job_probe.default_output_cols,
            nshards=nshards,
            store_uri=store_uri,
            checkpoint_every=checkpoint_every,
            checkpointing=checkpointing,
        )

    # Legacy pandas-internal runner.
    df = data if isinstance(data, pd.DataFrame) else backend.to_pandas(data)

    if checkpointing:
        if store_uri is None:
            raise ValueError("store_uri is required when checkpointing is enabled.")
        store: Any = ParquetShardStore(store_uri)
    else:
        store = InMemoryStore()
    cfg = RunnerConfig(id_col="_row_id", checkpoint_every=checkpoint_every)
    if nshards <= 1:
        out = await JobRunner(store, cfg).run(
            job_probe,
            df,
            input_col=input_col,
            output_cols=output_cols or job_probe.default_output_cols,
            output_mode=job_probe.output_mode,
        )
        return out if backend.name == "pandas" else backend.from_pandas(out)
    if not checkpointing:
        raise ValueError("Sharded execution requires checkpointing to be enabled.")
    indices = np.array_split(df.index, nshards)

    async def _one(i, idx):
        sub = df.loc[idx].copy()
        assert store_uri is not None
        su = store_uri.replace(".parquet", f"_shard{i}.parquet")
        store = ParquetShardStore(su)
        return await JobRunner(store, cfg).run(
            job_factory(),
            sub,
            input_col=input_col,
            output_cols=output_cols or job_probe.default_output_cols,
            output_mode=job_probe.output_mode,
        )

    parts = await asyncio.gather(*[_one(i, idx) for i, idx in enumerate(indices)])
    out = pd.concat(parts).sort_index()
    return out if backend.name == "pandas" else backend.from_pandas(out)


async def _run_job_ray(
    job_factory: Callable[[], Any],
    dataset: Any,
    *,
    input_col: str,
    output_cols: list[str] | None,
    batch_size: int,
    output_mode: Any,
) -> Any:
    """
    Native Ray execution path.

    For now, this bypasses ParquetShardStore checkpointing and uses Ray Dataset
    transforms to distribute LLM calls across workers.
    """
    try:
        import ray.data as rd  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Ray backend requires `ray[data]` to be installed.") from exc

    from domyn_swarm.jobs.base import OutputJoinMode
    from domyn_swarm.jobs.runner import _normalize_batch_outputs

    mode = output_mode
    if isinstance(mode, str):
        mode = OutputJoinMode(mode)
    if mode not in {OutputJoinMode.APPEND, OutputJoinMode.REPLACE, OutputJoinMode.IO_ONLY}:
        raise ValueError(f"Unsupported output_mode for ray backend: {mode}")

    ds = dataset

    def _process_batch(batch: pd.DataFrame) -> pd.DataFrame:
        job = job_factory()
        items = batch[input_col].tolist()
        out = job.transform_items(items)
        if inspect.isawaitable(out):
            out = asyncio.run(out)  # type: ignore[arg-type]
        rows, cols = _normalize_batch_outputs(out, output_cols)

        if cols is None:
            # dict outputs: join columns dynamically
            out_df = pd.DataFrame(rows)
        elif len(cols) == 1:
            out_df = pd.DataFrame({cols[0]: rows})
        else:
            # rows may be list/tuple per item, or list[list] per item
            out_df = pd.DataFrame({c: [r[i] for r in rows] for i, c in enumerate(cols)})

        if mode == OutputJoinMode.REPLACE:
            return out_df
        if mode == OutputJoinMode.IO_ONLY:
            keep = [input_col]
            return pd.concat([batch.loc[:, keep].reset_index(drop=True), out_df], axis=1)
        # APPEND
        return pd.concat([batch.reset_index(drop=True), out_df], axis=1)

    return ds.map_batches(
        _process_batch,
        batch_format="pandas",
        batch_size=batch_size,
    )
