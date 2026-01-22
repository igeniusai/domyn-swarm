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

import numpy as np

from domyn_swarm.checkpoint.store import ParquetShardStore
from domyn_swarm.helpers.data import compute_hash, parquet_hash
from domyn_swarm.helpers.io import load_dataframe, save_dataframe
from domyn_swarm.helpers.logger import setup_logger
from domyn_swarm.jobs import SwarmJob
from domyn_swarm.jobs.compat import run_job_unified  # base class
from domyn_swarm.jobs.runner import JobRunner, RunnerConfig

logger = setup_logger("domyn_swarm.jobs.run", level=logging.INFO)


def _load_cls(path: str) -> type[SwarmJob]:
    mod, cls = path.split(":")
    return getattr(importlib.import_module(mod), cls)


def _is_valid_parquet_file(path: Path) -> bool:
    """Return True if `path` can be opened as a Parquet file.

    Uses pyarrow footer parsing when available (fast, doesn't load full data).

    Args:
        path: Local parquet file path.

    Returns:
        True if the file appears to be a valid parquet file; False otherwise.
    """
    if not path.exists() or not path.is_file():
        return False
    if path.stat().st_size == 0:
        return False
    try:
        import pyarrow.parquet as pq  # type: ignore

        pq.ParquetFile(str(path))
        return True
    except Exception:
        try:
            # Fallback: attempt a minimal pandas read (may be slower than pyarrow footer parse).
            _ = load_dataframe(path, limit=1)
            return True
        except Exception:
            return False


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
        "--checkpoint-tag",
        type=str,
        default=None,
        help="An optional tag to be used when checkpointing is enabled. "
        "It will be used in place of the default hash-based tag.",
    )

    if not cli_args:
        return parser.parse_args()
    return parser.parse_args(args=cli_args)


def build_job_from_args(args) -> tuple[type[SwarmJob], dict]:
    cls_path = args.job_class or os.environ["JOB_CLASS"]
    model = args.model or os.environ["MODEL"]
    endpoint = args.endpoint or os.environ["ENDPOINT"]
    job_params = json.loads(args.job_kwargs or os.getenv("JOB_KWARGS", "{}"))
    kwargs = job_params.pop("kwargs", {})

    job_cls = _load_cls(cls_path)
    job_kwargs = {"endpoint": endpoint, "model": model, **job_params, **kwargs}
    return job_cls, job_kwargs


def _shard_filename(shard_id: int, nshards: int) -> str:
    width = max(1, len(str(nshards - 1)))
    return f"data-{shard_id:0{width}d}.parquet"


async def _amain(cli_args: list[str] | argparse.Namespace | None = None):
    args = parse_args(cli_args)
    in_path: Path = args.input_parquet or Path(os.environ["INPUT_PARQUET"])
    out_path: Path = args.output_parquet or Path(os.environ["OUTPUT_PARQUET"])
    job_cls, job_kwargs = build_job_from_args(args)
    df_in = load_dataframe(in_path, limit=args.limit)

    tag = args.checkpoint_tag or parquet_hash(in_path) + compute_hash(str(out_path))

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

    # Derive a default store URI from checkpoint dir (local file://); keeps legacy alive
    ckp_base = Path(args.checkpoint_dir) / f"{job_cls.__name__}_{tag}.parquet"
    store_uri = f"file://{ckp_base}"

    is_dir_output = out_path.is_dir() or out_path.suffix == ""
    input_col = job_kwargs.get("input_column_name", "messages")

    # For directory outputs + multiple shards, avoid materializing a giant concatenated DataFrame
    # (and avoid re-sharding it again in save_dataframe). Write each shard output directly.
    if nshards > 1 and is_dir_output:
        out_path.mkdir(parents=True, exist_ok=True)
        cfg = RunnerConfig(id_col="_row_id", checkpoint_every=args.checkpoint_interval)
        job_probe = make_job()
        resolved_output_cols = output_cols or job_probe.default_output_cols
        indices = np.array_split(df_in.index, nshards)

        for shard_id, idx in enumerate(indices):
            out_file = out_path / _shard_filename(shard_id, nshards)
            if out_file.exists() and not out_file.is_file():
                raise ValueError(f"Output shard path exists but is not a file: {out_file}")
            if _is_valid_parquet_file(out_file):
                logger.info("Output shard %s already exists and is valid; skipping.", out_file)
                continue
            sub = df_in.loc[idx].copy(deep=False)
            shard_store_uri = store_uri.replace(".parquet", f"_shard{shard_id}.parquet")
            store = ParquetShardStore(shard_store_uri)
            df_part = await JobRunner(store, cfg).run(
                make_job(),
                sub,
                input_col=input_col,
                output_cols=resolved_output_cols,
                output_mode=job_probe.output_mode,
            )
            df_part.to_parquet(out_file, index=False)
        return

    df_out = await run_job_unified(
        make_job,
        df_in,
        input_col=input_col,
        output_cols=output_cols,
        nshards=nshards,
        store_uri=store_uri,  # used by new-style
        checkpoint_every=args.checkpoint_interval,
        tag=tag,  # used by old-style
        checkpoint_dir=args.checkpoint_dir,  # used by old-style
    )

    save_dataframe(df_out, out_path, nshards=nshards)


def main(cli_args: list[str] | None = None):
    args = parse_args(cli_args or sys.argv[1:])
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
