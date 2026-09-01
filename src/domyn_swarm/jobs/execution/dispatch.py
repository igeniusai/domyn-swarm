# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
import warnings

from domyn_swarm.jobs.api.base import SwarmJob
from domyn_swarm.jobs.execution.arrow import _run_arrow
from domyn_swarm.jobs.execution.pandas import _run_pandas
from domyn_swarm.jobs.execution.polars import _run_polars
from domyn_swarm.jobs.execution.ray import run_ray_job
from domyn_swarm.jobs.io.backend import get_backend
from domyn_swarm.jobs.io.columns import _validate_required_id


def _has_new_api(job: SwarmJob) -> bool:
    """Return True if the job uses the new streaming API.

    Args:
        job: SwarmJob instance to inspect.

    Returns:
        True when the job is using the new API.
    """
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


def _is_overridden(obj: object, method: str, base: type) -> bool:
    """Return True if `method` is implemented by `type(obj)` or an intermediate base.

    Args:
        obj: Object instance to inspect.
        method: Method name to check.
        base: Base class to compare against.

    Returns:
        True if the method is overridden by a subclass.
    """
    cls = type(obj)
    for c in cls.mro():
        if method in c.__dict__:
            return c is not base
    return False


def resolve_job_api(job: SwarmJob) -> str:
    """Resolve the job API version label for a SwarmJob.

    Args:
        job: SwarmJob instance to inspect.

    Returns:
        "new" for streaming API jobs, otherwise "old".
    """
    if getattr(type(job), "api_version", 1) >= 2:
        return "new"
    if _is_overridden(job, "transform_items", SwarmJob) or _is_overridden(
        job, "transform_streaming", SwarmJob
    ):
        return "new"
    if _is_overridden(job, "transform", SwarmJob) or _is_overridden(job, "run", SwarmJob):
        return "old"
    return "old"


_VALID_ENGINES = ("pandas", "arrow", "polars", "ray")


def resolve_engine(
    *,
    engine: str | None,
    data_backend: str | None,
    runner: str,
    warn: bool = False,
) -> str:
    """Resolve the execution engine from the new or deprecated selectors.

    Args:
        engine: The engine name, when given explicitly. Takes precedence.
        data_backend: Resolved backend name (the value that already accounts for
            the job's own `data_backend` default), used only as part of the
            deprecated `runner`/`data_backend` pair when `engine` is not given.
        runner: Deprecated runner selector.
        warn: Whether to emit a DeprecationWarning when falling back to the pair.

    Returns:
        One of "pandas", "arrow", "polars" or "ray".

    Raises:
        ValueError: If `engine` is not a supported name.
    """
    if engine is not None:
        if engine not in _VALID_ENGINES:
            raise ValueError(f"Unknown engine {engine!r}; choose one of {_VALID_ENGINES}.")
        return engine

    if warn:
        warnings.warn(
            "`runner` is deprecated; pass `engine=` instead (one of "
            f"{_VALID_ENGINES}). `data_backend` is not deprecated and keeps "
            "choosing the IO/conversion backend, but no longer affects routing. "
            "`runner` will be removed in a future release.",
            DeprecationWarning,
            stacklevel=3,
        )

    if data_backend == "ray":
        return "ray"
    if data_backend == "polars" and runner == "arrow":
        return "polars"
    if runner == "arrow":
        return "arrow"
    return "pandas"


# Engines whose implementation assumes a specific resolved data backend and
# cannot be handed anything else: `run_ray_job` takes `data` directly and
# expects a Ray Dataset, and `_run_polars` expects a Polars DataFrame/LazyFrame.
# `engine` decouples routing from `data_backend`, so this is the guard that
# stops a caller from naming a combination that would fail deep inside Ray
# Data or Polars instead of at the call site.
_ENGINE_REQUIRED_BACKEND = {"ray": "ray", "polars": "polars"}


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
    engine: str | None = None,
    ray_address: str | None = None,
    output_path: Path | None = None,
    shard_output: bool = False,
    shard_mode: str = "id",
    global_resume: bool = False,
) -> Any:
    """Run a SwarmJob with backend-aware execution and checkpointing.

    Args:
        job_factory: Callable that returns a SwarmJob instance.
        data: Input dataset in backend-native form (pandas, polars, arrow, ray dataset, etc.).
        input_col: Column name to read inputs from.
        output_cols: Optional list of output column names (None for dict outputs).
        nshards: Number of shards to split input into for non-ray execution.
        store_uri: Base checkpoint store URI (required when checkpointing is enabled).
        checkpoint_every: Flush interval in items.
        data_backend: Backend name override (defaults to the job's `data_backend` or "pandas").
            Continues to select the `DataBackend` used for IO and conversions; it no
            longer affects the routing decision, which is made by `engine`.
        native_backend: Override for native execution (required for ray).
        checkpointing: Whether to read/write checkpoint state.
        runner: Deprecated. Use `engine` instead. Runner implementation to use for
            non-ray backends ("pandas" or "arrow"), consulted only when `engine`
            is not given.
        engine: Execution engine to use ("pandas", "arrow", "polars" or "ray").
            Takes precedence over the deprecated `runner`/`data_backend` pair.
            Defaults to the value resolved from that pair when not given.
        ray_address: Optional Ray cluster address (only used for ray backend).
        output_path: Optional output path used to enable direct shard writes when using
            the pandas runner and directory outputs.
        shard_output: If True and output_path is a directory, write one parquet file per
            shard directly from checkpoint outputs and return None, rather than assembling
            the merged result in memory and returning it. Read by the pandas and polars
            engines; ignored by the Arrow and Ray engines. On the pandas engine, this also
            makes `global_resume` a no-op: shards are sharded from the full input and each
            shard's own checkpoint store already supplies resume, which is equivalent to
            global resume as long as `nshards` stays fixed across resumes of the same job.
        shard_mode: Sharding strategy ("id" for stable id hashing, "index" for legacy order).
        global_resume: Resume by filtering inputs with global done ids across shards. Ignored
            on the pandas engine when `shard_output` is True (see `shard_output`).

    Returns:
        Backend-native result for non-ray runs, or the Ray runner result.
        Returns None when the pandas or polars paths write outputs directly to a directory.

    Raises:
        TypeError: If the job does not implement the streaming API.
        ValueError: If required id columns are missing, if checkpointing is
            misconfigured, or if the resolved engine requires a data backend
            other than the one resolved (e.g. `engine="ray"` with a non-ray
            backend, or `engine="polars"` with a non-polars backend).
        RuntimeError: If the backend cannot be resolved.
    """
    job_probe = job_factory()
    _ensure_new_api(job_probe)
    backend_name = _resolve_backend_name(job_probe, data_backend)
    id_col, require_id = _resolve_id_column(job_probe)
    backend = get_backend(backend_name)

    resolved_engine = resolve_engine(
        engine=engine,
        data_backend=backend.name,
        runner=runner,
        warn=engine is None and runner != "pandas",
    )
    required_backend = _ENGINE_REQUIRED_BACKEND.get(resolved_engine)
    if required_backend is not None and backend.name != required_backend:
        raise ValueError(
            f"engine={resolved_engine!r} requires the {required_backend!r} data backend, "
            f"but the resolved backend is {backend.name!r}. Pass "
            f"data_backend={required_backend!r} (or declare it on the job)."
        )

    if resolved_engine == "ray":
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

    if resolved_engine == "polars":
        return await _run_polars(
            job_factory=job_factory,
            backend=backend,
            data=data,
            input_col=input_col,
            output_cols=output_cols or job_probe.default_output_cols,
            id_col=id_col,
            require_id=require_id,
            nshards=nshards,
            shard_mode=shard_mode,
            global_resume=global_resume,
            store_uri=store_uri,
            checkpoint_every=checkpoint_every,
            checkpointing=checkpointing,
            output_path=output_path,
            shard_output=shard_output,
        )

    if resolved_engine == "arrow":
        result = await _run_arrow(
            job_factory=job_factory,
            backend=backend,
            data=data,
            input_col=input_col,
            output_cols=output_cols or job_probe.default_output_cols,
            id_col=id_col,
            require_id=require_id,
            nshards=nshards,
            shard_mode=shard_mode,
            global_resume=global_resume,
            store_uri=store_uri,
            checkpoint_every=checkpoint_every,
            checkpointing=checkpointing,
        )
        return backend.from_arrow(result)

    return await _run_pandas(
        job_factory=job_factory,
        job_probe=job_probe,
        backend=backend,
        data=data,
        input_col=input_col,
        output_cols=output_cols,
        id_col=id_col,
        require_id=require_id,
        nshards=nshards,
        shard_mode=shard_mode,
        global_resume=global_resume,
        store_uri=store_uri,
        checkpoint_every=checkpoint_every,
        checkpointing=checkpointing,
        output_path=output_path,
        shard_output=shard_output,
    )
