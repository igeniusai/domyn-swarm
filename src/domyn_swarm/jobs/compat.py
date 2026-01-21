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
import math
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from domyn_swarm.checkpoint.store import InMemoryStore, ParquetShardStore
from domyn_swarm.data import BackendError, get_backend
from domyn_swarm.data.backends.base import DataBackend
from domyn_swarm.jobs.arrow_runner import run_arrow_job
from domyn_swarm.jobs.base import SwarmJob
from domyn_swarm.jobs.polars_runner import run_polars_job
from domyn_swarm.jobs.ray_runner import run_ray_job
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
    id_col: str,
    require_id: bool,
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
        id_col: Column name used for stable row ids.
        require_id: Whether id_col must already exist in the input.
        nshards: Number of shards to split the input into.
        store_uri: Base checkpoint store URI.
        checkpoint_every: Flush interval in items.
        checkpointing: Whether checkpointing is enabled.

    Returns:
        Arrow table containing job outputs (and inputs depending on output mode).
    """
    table = data if isinstance(data, pa.Table) else backend.to_arrow(data)
    if require_id and id_col not in table.column_names:
        raise ValueError(f"Input table missing required id column '{id_col}'.")
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
            id_col=id_col,
        )
    if not checkpointing:
        raise ValueError("Sharded execution requires checkpointing to be enabled.")

    if id_col not in table.column_names:
        table = _ensure_arrow_id(table, id_col)
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
            id_col=id_col,
        )

    parts = await asyncio.gather(*[_one(i, idx) for i, idx in enumerate(indices)])
    return pa.concat_tables(parts)


def _has_new_api(job: SwarmJob) -> bool:
    return resolve_job_api(job) == "new"


def _ensure_new_api(job: SwarmJob) -> None:
    """Validate that the job exposes the streaming API.

    Args:
        job: SwarmJob instance to validate.

    Raises:
        TypeError: If the job does not implement the streaming API.
    """
    if not _has_new_api(job):
        raise TypeError(
            "Job must implement the streaming API (transform_items/transform_streaming)."
        )


def _resolve_backend_name(job: SwarmJob, data_backend: str | None) -> str:
    """Resolve the backend name for a job run.

    Args:
        job: SwarmJob instance providing defaults.
        data_backend: Optional backend override.

    Returns:
        Backend name to use for IO and conversions.
    """
    return data_backend or getattr(job, "data_backend", None) or "pandas"


def _resolve_id_column(job: SwarmJob) -> tuple[str, bool]:
    """Resolve the id column name and whether it is required.

    Args:
        job: SwarmJob instance with optional id_column_name.

    Returns:
        Tuple of (id_col, require_id).

    Raises:
        ValueError: If the id_column_name is invalid.
    """
    raw_id_col = getattr(job, "id_column_name", None)
    if raw_id_col is not None:
        if not isinstance(raw_id_col, str):
            raise ValueError("id_column_name must be a string when provided.")
        if not raw_id_col.strip():
            raise ValueError("id_column_name must be a non-empty string.")
    return raw_id_col or "_row_id", raw_id_col is not None


def _get_backend(backend_name: str) -> DataBackend:
    """Resolve a data backend by name.

    Args:
        backend_name: Backend name to load.

    Returns:
        Loaded DataBackend instance.

    Raises:
        RuntimeError: If the backend cannot be resolved.
    """
    try:
        return get_backend(backend_name)
    except BackendError as exc:
        raise RuntimeError(str(exc)) from exc


def _column_names_from_collect_schema(data: Any) -> list[str] | None:
    """Try to read column names via a `collect_schema()` method.

    This avoids triggering expensive schema resolution on objects like polars LazyFrame.

    Args:
        data: Input dataset-like object.

    Returns:
        List of column names if available, otherwise None.
    """
    collect_schema = getattr(data, "collect_schema", None)
    if not callable(collect_schema):
        return None
    try:
        schema = collect_schema()
        names = getattr(schema, "names", None)
        if callable(names):
            names = names()
        if names is None:
            return None
        if isinstance(names, list):
            return names
        if isinstance(names, tuple):
            return list(names)
        return None
    except Exception:
        return None


def _column_names_from_columns_attr(data: Any) -> list[str] | None:
    """Try to read column names from a `.columns` attribute.

    Args:
        data: Input dataset-like object.

    Returns:
        List of column names if available, otherwise None.
    """
    if not hasattr(data, "columns"):
        return None
    try:
        return list(data.columns)
    except TypeError:
        return None


def _column_names_from_schema(data: Any) -> list[str] | None:
    """Try to read column names from a `.schema()` method.

    Args:
        data: Input dataset-like object.

    Returns:
        List of column names if available, otherwise None.
    """
    if not hasattr(data, "schema"):
        return None
    try:
        schema = data.schema()
    except Exception:
        return None
    names: list[str] | None = getattr(schema, "names", None)
    if callable(names):
        names = names()
    if names is None:
        try:
            names = [field.name for field in schema]
        except TypeError:
            names = []
    return names


def _validate_required_id(data: Any, id_col: str) -> None:
    """Validate that the input data contains the required id column.

    Args:
        data: Input dataset or DataFrame.
        id_col: Required column name.

    Raises:
        ValueError: If the column is missing or cannot be validated.
    """
    names = (
        _column_names_from_collect_schema(data)
        or _column_names_from_columns_attr(data)
        or _column_names_from_schema(data)
    )
    if names is None:
        raise ValueError(f"Cannot validate id column on data type: {type(data)!r}")
    if id_col not in names:
        raise ValueError(f"Input data missing required id column '{id_col}'.")


def _resolve_ray_native(native_backend: bool | None) -> bool:
    """Resolve and validate native execution for the Ray backend.

    Args:
        native_backend: Optional native backend override.

    Returns:
        True when native execution is enabled.

    Raises:
        ValueError: If native execution is disabled for Ray.
    """
    native = bool(native_backend) if native_backend is not None else True
    if not native:
        raise ValueError("Ray backend requires native execution (native_backend=True).")
    return native


def _build_checkpoint_store(
    *,
    checkpointing: bool,
    store_uri: str | None,
) -> InMemoryStore | ParquetShardStore:
    """Construct a checkpoint store based on flags.

    Args:
        checkpointing: Whether checkpointing is enabled.
        store_uri: Store URI for checkpointing.

    Returns:
        Checkpoint store implementation.

    Raises:
        ValueError: If checkpointing is enabled without a store URI.
    """
    if checkpointing:
        if store_uri is None:
            raise ValueError("store_uri is required when checkpointing is enabled.")
        return ParquetShardStore(store_uri)
    return InMemoryStore()


async def _run_non_ray_polars(
    *,
    job_factory: Callable[[], Any],
    backend: DataBackend,
    data: Any,
    input_col: str,
    output_cols: list[str] | None,
    id_col: str,
    require_id: bool,
    nshards: int,
    store_uri: str | None,
    checkpoint_every: int,
    checkpointing: bool,
) -> Any:
    """Run the polars execution path for non-ray backends.

    Args:
        job_factory: Callable producing a SwarmJob instance.
        data: Polars DataFrame or LazyFrame.
        input_col: Input column name in the dataset.
        output_cols: Output columns to produce (None for dict outputs).
        id_col: Column name used for stable row ids.
        require_id: Whether id_col must already exist in the input.
        nshards: Number of shards to split the input into.
        store_uri: Base checkpoint store URI.
        checkpoint_every: Flush interval in items.
        checkpointing: Whether checkpointing is enabled.

    Returns:
        Polars DataFrame containing job outputs.
    """
    import polars as pl

    if isinstance(data, pd.DataFrame):
        data = pl.from_pandas(data)
    elif isinstance(data, pa.Table):
        data = pl.from_arrow(data)

    if not isinstance(data, pl.DataFrame | pl.LazyFrame):
        raise ValueError("Polars runner expects a polars DataFrame or LazyFrame.")
    if require_id:
        _validate_required_id(data, id_col)
    if checkpointing and store_uri is None:
        raise ValueError("store_uri is required when checkpointing is enabled.")

    if nshards <= 1:
        return await run_polars_job(
            job_factory,
            backend,
            data,
            input_col=input_col,
            output_cols=output_cols,
            store_uri=store_uri,
            checkpoint_every=checkpoint_every,
            checkpointing=checkpointing,
            id_col=id_col,
        )
    if not checkpointing:
        raise ValueError("Sharded execution requires checkpointing to be enabled.")

    if id_col not in data.columns:
        data = data.with_row_index(id_col)

    if isinstance(data, pl.LazyFrame):
        data = data.collect()

    total_rows = data.height
    if total_rows == 0:
        return await run_polars_job(
            job_factory,
            backend,
            data,
            input_col=input_col,
            output_cols=output_cols,
            store_uri=store_uri,
            checkpoint_every=checkpoint_every,
            checkpointing=checkpointing,
            id_col=id_col,
        )

    chunk_size = math.ceil(total_rows / nshards)

    async def _one(i: int, start: int):
        assert store_uri is not None
        sub = data.slice(start, chunk_size)
        su = store_uri.replace(".parquet", f"_shard{i}.parquet")
        return await run_polars_job(
            job_factory,
            backend,
            sub,
            input_col=input_col,
            output_cols=output_cols,
            store_uri=su,
            checkpoint_every=checkpoint_every,
            checkpointing=checkpointing,
            id_col=id_col,
        )

    tasks = []
    for shard_id in range(nshards):
        start = shard_id * chunk_size
        if start >= total_rows:
            break
        tasks.append(_one(shard_id, start))
    parts = await asyncio.gather(*tasks)
    return pl.concat(parts, how="vertical")


async def _run_non_ray_pandas(
    *,
    job_factory: Callable[[], Any],
    job_probe: SwarmJob,
    backend: DataBackend,
    data: Any,
    input_col: str,
    output_cols: list[str] | None,
    id_col: str,
    require_id: bool,
    nshards: int,
    store_uri: str | None,
    checkpoint_every: int,
    checkpointing: bool,
) -> Any:
    """Run the pandas-backed execution path for non-ray backends.

    Args:
        job_factory: Callable producing a SwarmJob instance.
        job_probe: Probe job instance for defaults.
        backend: Data backend used for conversion.
        data: Backend-native data or DataFrame.
        input_col: Input column name.
        output_cols: Output column names (None for dict outputs).
        id_col: Column name used for stable row ids.
        require_id: Whether id_col must already exist in the input.
        nshards: Number of shards to split the input into.
        store_uri: Base checkpoint store URI.
        checkpoint_every: Flush interval in items.
        checkpointing: Whether checkpointing is enabled.

    Returns:
        Job results in backend-native output form.
    """
    df = data if isinstance(data, pd.DataFrame) else backend.to_pandas(data)
    if require_id:
        _validate_required_id(df, id_col)

    cfg = RunnerConfig(id_col=id_col, checkpoint_every=checkpoint_every)
    if nshards <= 1:
        store = _build_checkpoint_store(checkpointing=checkpointing, store_uri=store_uri)
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
    data_backend: str | None = None,
    native_backend: bool | None = None,
    checkpointing: bool = True,
    runner: str = "pandas",
    ray_address: str | None = None,
) -> Any:
    job_probe = job_factory()
    _ensure_new_api(job_probe)
    backend_name = _resolve_backend_name(job_probe, data_backend)
    id_col, require_id = _resolve_id_column(job_probe)
    backend = _get_backend(backend_name)

    if backend.name == "ray":
        _resolve_ray_native(native_backend)
        if not require_id:
            raise ValueError("Ray backend requires a user-provided id column (use --id-column).")
        _validate_required_id(data, id_col)
        return await run_ray_job(
            job_factory,
            data,
            input_col=input_col,
            output_cols=output_cols or job_probe.default_output_cols,
            batch_size=getattr(job_probe, "native_batch_size", None) or checkpoint_every,
            output_mode=job_probe.output_mode,
            id_col=id_col,
            store_uri=store_uri,
            checkpointing=checkpointing,
            compact=True,
            ray_address=ray_address,
        )

    if runner == "arrow" and backend.name == "polars":
        return await _run_non_ray_polars(
            job_factory=job_factory,
            backend=backend,
            data=data,
            input_col=input_col,
            output_cols=output_cols or job_probe.default_output_cols,
            id_col=id_col,
            require_id=require_id,
            nshards=nshards,
            store_uri=store_uri,
            checkpoint_every=checkpoint_every,
            checkpointing=checkpointing,
        )

    if runner == "arrow":
        result = await _run_non_ray_arrow(
            job_factory=job_factory,
            backend=backend,
            data=data,
            input_col=input_col,
            output_cols=output_cols or job_probe.default_output_cols,
            id_col=id_col,
            require_id=require_id,
            nshards=nshards,
            store_uri=store_uri,
            checkpoint_every=checkpoint_every,
            checkpointing=checkpointing,
        )
        return backend.from_arrow(result)

    return await _run_non_ray_pandas(
        job_factory=job_factory,
        job_probe=job_probe,
        backend=backend,
        data=data,
        input_col=input_col,
        output_cols=output_cols,
        id_col=id_col,
        require_id=require_id,
        nshards=nshards,
        store_uri=store_uri,
        checkpoint_every=checkpoint_every,
        checkpointing=checkpointing,
    )
