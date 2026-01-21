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

from __future__ import annotations

import asyncio
import inspect
import os
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import pandas as pd
from ulid import ULID

from domyn_swarm.jobs.base import OutputJoinMode
from domyn_swarm.jobs.runner import _normalize_batch_outputs


def _file_uri_to_local_path(uri: str) -> Path:
    """Convert a `file://` URI into a local filesystem path.

    Args:
        uri: A `file://...` URI (or a plain path).

    Returns:
        Local filesystem path.

    Raises:
        ValueError: If the URI scheme is not supported.
    """
    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        return Path(unquote(parsed.path or uri))
    raise ValueError(f"Unsupported URI scheme for ray checkpointing: {parsed.scheme!r}")


def _ray_checkpoint_base_dir(store_uri: str) -> Path:
    """Resolve the base directory for Ray checkpoint/resume from store_uri.

    This accepts the existing `store_uri` convention used by non-ray backends
    (e.g., `file:///.../Foo.parquet`) and maps it to a directory path
    (`.../Foo`) where Ray can write dataset directories.

    Args:
        store_uri: Store URI used by other runners for checkpointing.

    Returns:
        Base directory to use for Ray checkpoint runs.
    """
    path = _file_uri_to_local_path(store_uri)
    return path.with_suffix("") if path.suffix else path


def _ray_list_run_dirs(runs_dir: Path) -> list[Path]:
    """List checkpoint run directories for the Ray output-as-checkpoint strategy.

    Args:
        runs_dir: Directory containing per-run output directories.

    Returns:
        Sorted list of run directories.
    """
    if not runs_dir.exists():
        return []
    return sorted([p for p in runs_dir.iterdir() if p.is_dir()])


def _ray_load_done_ids(rd: Any, *, runs_dir: Path, id_col: str) -> set[Any]:
    """Load completed ids from existing Ray checkpoint run directories.

    Note:
        This currently collects ids to the driver via `take_all()`. For very large outputs,
        this should be replaced with a fully distributed anti-join strategy.

    Args:
        rd: `ray.data` module.
        runs_dir: Directory containing per-run output directories.
        id_col: Column name containing stable ids.

    Returns:
        Set of ids already present in checkpoint outputs.
    """
    run_dirs = _ray_list_run_dirs(runs_dir)
    if not run_dirs:
        return set()
    done_ids_ds = rd.read_parquet([p.as_posix() for p in run_dirs], columns=[id_col])
    done_rows = done_ids_ds.take_all()
    return {row[id_col] for row in done_rows if id_col in row}


def _ray_filter_done_ids(ds: Any, *, done_ids: set[Any], id_col: str) -> Any:
    """Filter a Ray Dataset to exclude rows already present in checkpoints.

    Args:
        ds: Input Ray Dataset.
        done_ids: Set of ids to exclude.
        id_col: Column name containing stable ids.

    Returns:
        Filtered Ray Dataset.
    """
    if not done_ids:
        return ds
    done_ids_list = list(done_ids)

    def _filter_done(batch: pd.DataFrame) -> pd.DataFrame:
        return batch.loc[~batch[id_col].isin(done_ids_list)]

    return ds.map_batches(_filter_done, batch_format="pandas")


def _ray_write_run_outputs(out_ds: Any, *, runs_dir: Path) -> Path:
    """Write a single run's outputs to a new checkpoint directory.

    Args:
        out_ds: Ray Dataset containing outputs for this run.
        runs_dir: Directory containing per-run output directories.

    Returns:
        The created run directory path.
    """
    run_id = str(ULID()).lower()
    run_dir = runs_dir / f"run-{run_id}"
    out_ds.write_parquet(run_dir.as_posix())
    return run_dir


def _ray_compact_runs(rd: Any, *, runs_dir: Path) -> Any:
    """Compact all checkpoint run directories into a single Ray Dataset.

    Args:
        rd: `ray.data` module.
        runs_dir: Directory containing per-run output directories.

    Returns:
        Ray Dataset produced by reading all run directories.
    """
    run_dirs = _ray_list_run_dirs(runs_dir)
    return rd.read_parquet([p.as_posix() for p in run_dirs]) if run_dirs else rd.from_items([])


def _ensure_ray_initialized(ray: Any, *, ray_address: str | None = None) -> None:
    """Ensure Ray is initialized, preferring an existing cluster when available.

    This runner is typically executed inside an environment where a Ray cluster is already
    running. We connect to that cluster via an explicit address (CLI/env), and fail fast
    if no address is provided to avoid accidentally starting a local Ray runtime.

    Args:
        ray: Imported `ray` module.
        ray_address: Cluster address passed via CLI/env.
    """
    if ray.is_initialized():
        return

    address = (
        ray_address or os.environ.get("DOMYN_SWARM_RAY_ADDRESS") or os.environ.get("RAY_ADDRESS")
    )
    if not address:
        raise ValueError(
            "Ray backend requires an explicit ray address (set --ray-address, "
            "DOMYN_SWARM_RAY_ADDRESS, or RAY_ADDRESS)."
        )
    ray.init(address=address, ignore_reinit_error=True, log_to_driver=False)


async def run_ray_job(
    job_factory: Any,
    dataset: Any,
    *,
    input_col: str,
    output_cols: list[str] | None,
    batch_size: int,
    output_mode: Any,
    id_col: str,
    store_uri: str | None,
    checkpointing: bool,
    compact: bool,
    ray_address: str | None = None,
) -> Any:
    """Run a SwarmJob using the Ray native execution path with output-as-checkpoint resume.

    Strategy: output-as-checkpoint.
      - Each run writes to a new checkpoint directory under a deterministic base dir.
      - Resume loads existing checkpoint outputs, extracts done ids, and filters the input dataset.
      - Final output is produced by compacting all run directories into a single dataset.

    Args:
        job_factory: Callable producing a SwarmJob instance.
        dataset: Ray Dataset input.
        input_col: Column name containing job inputs.
        output_cols: Output column names (None for dict outputs).
        batch_size: Ray batch size for map_batches.
        output_mode: OutputJoinMode (APPEND/REPLACE/IO_ONLY).
        id_col: Required stable id column name.
        store_uri: Base URI used for checkpointing (file:// only for now).
        checkpointing: Whether to read/write checkpoint state.
        compact: Whether to compact checkpoint outputs into a single final dataset.

    Returns:
        Ray Dataset representing job results.
    """
    try:
        import ray
        import ray.data as rd
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Ray backend requires `ray[data]` to be installed.") from exc

    _ensure_ray_initialized(ray, ray_address=ray_address)

    mode = output_mode
    if isinstance(mode, str):
        mode = OutputJoinMode(mode)
    if mode not in {OutputJoinMode.APPEND, OutputJoinMode.REPLACE, OutputJoinMode.IO_ONLY}:
        raise ValueError(f"Unsupported output_mode for ray backend: {mode}")

    ds = dataset

    runs_dir: Path | None = None
    if checkpointing:
        if store_uri is None:
            raise ValueError("store_uri is required when checkpointing is enabled.")
        checkpoint_base = _ray_checkpoint_base_dir(store_uri)
        runs_dir = checkpoint_base / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        done_ids = _ray_load_done_ids(rd, runs_dir=runs_dir, id_col=id_col)
        ds = _ray_filter_done_ids(ds, done_ids=done_ids, id_col=id_col)

    def _process_batch(batch: pd.DataFrame) -> pd.DataFrame:
        job = job_factory()
        items = batch[input_col].tolist()
        out = job.transform_items(items)
        if inspect.isawaitable(out):
            out = asyncio.run(out)  # type: ignore[arg-type]
        rows, cols = _normalize_batch_outputs(out, output_cols)

        if cols is None:
            out_df = pd.DataFrame(rows)
        elif len(cols) == 1:
            out_df = pd.DataFrame({cols[0]: rows})
        else:
            out_df = pd.DataFrame({c: [r[i] for r in rows] for i, c in enumerate(cols)})

        if mode == OutputJoinMode.REPLACE:
            return pd.concat([batch.loc[:, [id_col]].reset_index(drop=True), out_df], axis=1)
        if mode == OutputJoinMode.IO_ONLY:
            keep = [id_col, input_col]
            return pd.concat([batch.loc[:, keep].reset_index(drop=True), out_df], axis=1)
        return pd.concat([batch.reset_index(drop=True), out_df], axis=1)

    out_ds = ds.map_batches(
        _process_batch,
        batch_format="pandas",
        batch_size=batch_size,
    )

    if not checkpointing:
        return out_ds

    assert runs_dir is not None
    _ray_write_run_outputs(out_ds, runs_dir=runs_dir)

    if not compact:
        return out_ds

    return _ray_compact_runs(rd, runs_dir=runs_dir)
