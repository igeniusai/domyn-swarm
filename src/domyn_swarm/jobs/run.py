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

# domyn_swarm/jobs/run.py
import argparse
import asyncio
import importlib
import json
import logging
import os
from pathlib import Path
import sys

from ulid import ULID

from domyn_swarm.data import BackendError, get_backend
from domyn_swarm.helpers.data import compute_hash, parquet_hash
from domyn_swarm.helpers.logger import setup_logger
from domyn_swarm.jobs import SwarmJob
from domyn_swarm.jobs.compat import run_job_unified  # base class

logger = setup_logger("domyn_swarm.jobs.run", level=logging.INFO)


def _load_cls(path: str) -> type[SwarmJob]:
    mod, cls = path.split(":")
    return getattr(importlib.import_module(mod), cls)


def _write_result(
    backend,
    result,
    out_path: Path,
    nshards: int,
    backend_write_kwargs: dict,
    runner_choice: str,
):
    """Write a job result using the selected backend.

    Args:
        backend: Data backend used for writing.
        result: Job result in backend-native or Arrow form.
        out_path: Output path for the result.
        nshards: Number of shards to write (backend-dependent).
        backend_write_kwargs: Extra kwargs forwarded to backend write().
        runner_choice: Runner implementation name (pandas or arrow).
    """

    if runner_choice == "arrow":
        result = backend.from_arrow(result)

    backend.write(result, out_path, nshards=nshards, **backend_write_kwargs)


def parse_args(cli_args=None):
    if isinstance(cli_args, argparse.Namespace):
        return cli_args

    parser = argparse.ArgumentParser(description="Run a SwarmJob on a Parquet input.")

    parser.add_argument("--job-class", type=str, help="Job class path in format 'pkg.module:Class'")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--input-parquet", type=Path, help="Path to input Parquet file")
    parser.add_argument("--output-parquet", type=Path, help="Path to output Parquet file")
    parser.add_argument("--endpoint", type=str, help="Endpoint URL")
    parser.add_argument("--job-kwargs", type=str, default="{}", help="Extra JSON string kwargs")
    parser.add_argument(
        "--id-column",
        "--id-col",
        type=str,
        default=None,
        help="Optional column name used for stable row ids.",
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        default=1,
        help="How many threads should be used when executing this job",
    )
    parser.add_argument("--limit", default=None, type=int, required=False)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(".checkpoints"),
        help="Directory to store checkpoint files",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=16,
        help="How often to checkpoint progress (in records)",
    )
    parser.add_argument(
        "--no-resume",
        "--ignore-checkpoints",
        action="store_true",
        help="Ignore existing checkpoints for this run (forces recompute).",
    )
    parser.add_argument(
        "--no-checkpointing",
        action="store_true",
        help="Disable checkpointing entirely (no read/write checkpoint state).",
    )
    parser.add_argument(
        "--runner",
        choices=["pandas", "arrow"],
        default="pandas",
        help="Runner implementation to use for non-ray backends.",
    )

    if not cli_args:
        return parser.parse_args()
    return parser.parse_args(args=cli_args)


def build_job_from_args(args) -> tuple[type[SwarmJob], dict]:
    cls_path = args.job_class or os.environ["JOB_CLASS"]
    model = args.model or os.environ["MODEL"]
    endpoint = args.endpoint or os.environ["ENDPOINT"]
    job_params = json.loads(args.job_kwargs or os.getenv("JOB_KWARGS", "{}"))
    if getattr(args, "id_column", None):
        job_params["id_column_name"] = args.id_column
    kwargs = job_params.pop("kwargs", {})

    job_cls = _load_cls(cls_path)
    job_kwargs = {"endpoint": endpoint, "model": model, **job_params, **kwargs}
    return job_cls, job_kwargs


async def _amain(cli_args: list[str] | argparse.Namespace | None = None):
    args = parse_args(cli_args)
    in_path: Path = args.input_parquet or Path(os.environ["INPUT_PARQUET"])
    out_path: Path = args.output_parquet or Path(os.environ["OUTPUT_PARQUET"])
    job_cls, job_kwargs = build_job_from_args(args)
    backend_name = job_kwargs.get("data_backend")
    backend_read_kwargs = job_kwargs.get("backend_read_kwargs") or {}
    backend_write_kwargs = job_kwargs.get("backend_write_kwargs") or {}
    native_backend = job_kwargs.get("native_backend")
    runner_choice = getattr(args, "runner", "pandas")

    if not isinstance(backend_read_kwargs, dict):
        raise ValueError("backend_read_kwargs must be a dict if provided")
    if not isinstance(backend_write_kwargs, dict):
        raise ValueError("backend_write_kwargs must be a dict if provided")

    try:
        backend = get_backend(backend_name)
    except BackendError as exc:
        raise RuntimeError(str(exc)) from exc

    if native_backend is None and backend.name == "ray":
        native_backend = True

    data_in = backend.read(in_path, limit=args.limit, **backend_read_kwargs)
    tag = parquet_hash(in_path) + compute_hash(str(out_path))

    def make_job():
        return job_cls(**job_kwargs)

    # Resolve output columns for new-style (fallback to legacy CLI flag)
    output_cols = None
    if hasattr(job_cls, "output_cols"):
        o = job_cls.output_cols  # type: ignore[attr-defined]
        output_cols = o if isinstance(o, list) else [o]
    elif "output_cols" in job_kwargs:
        o = job_kwargs["output_cols"]
        output_cols = o if isinstance(o, list) else [o]

    # Map legacy --nthreads to shards
    nshards = getattr(args, "nthreads", 1)

    checkpointing = not args.no_checkpointing
    no_resume = args.no_resume
    if not checkpointing and no_resume:
        raise ValueError("--no-resume cannot be combined with --no-checkpointing.")

    # Derive a default store URI from checkpoint dir (local file://); keeps legacy alive.
    # If --no-resume is set, generate a fresh store name to avoid mixing with older runs.
    store_uri = None
    if checkpointing:
        store_name = f"{job_cls.__name__}_{tag}"
        if no_resume:
            store_name = f"{store_name}_{str(ULID()).lower()}"
        ckp_base = Path(args.checkpoint_dir) / f"{store_name}.parquet"
        store_uri = f"file://{ckp_base}"

    result = await run_job_unified(
        make_job,
        data_in,
        input_col=job_kwargs.get("input_column_name", "messages"),
        output_cols=output_cols,
        nshards=nshards,
        store_uri=store_uri,  # used by new-style
        checkpoint_every=args.checkpoint_interval,
        data_backend=backend.name,
        native_backend=native_backend,
        checkpointing=checkpointing,
        runner=runner_choice,
    )

    _write_result(backend, result, out_path, nshards, backend_write_kwargs, runner_choice)


def main(cli_args: list[str] | None = None):
    args = parse_args(cli_args or sys.argv[1:])
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
