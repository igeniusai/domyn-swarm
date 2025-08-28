# domyn_swarm/jobs/run.py
import argparse
import asyncio
import importlib
import json
import logging
import os
import sys
import threading
import warnings
from pathlib import Path
from typing import Optional, Type, Union

import numpy as np
import pandas as pd

from domyn_swarm.helpers.data import compute_hash, parquet_hash
from domyn_swarm.helpers.io import load_dataframe, save_dataframe
from domyn_swarm.helpers.logger import setup_logger
from domyn_swarm.jobs import SwarmJob
from domyn_swarm.jobs.compat import run_job_unified  # base class

logger = setup_logger("domyn_swarm.jobs.run", level=logging.INFO)


def _load_cls(path: str) -> type[SwarmJob]:
    mod, cls = path.split(":")
    return getattr(importlib.import_module(mod), cls)


async def run_swarm_in_threads(
    df: pd.DataFrame,
    job_cls: Type["SwarmJob"],
    *,
    job_kwargs: dict,
    tag: str,
    checkpoint_dir: Union[str, Path] = ".checkpoints",
    num_threads: Optional[int] = None,
) -> pd.DataFrame:
    """
    Runs SwarmJob across multiple threads with individual asyncio event loops.

    Each thread gets a shard of the input DataFrame and runs the full pipeline
    asynchronously. Checkpoints and results are kept per-thread and merged at the end.

    Parameters:
        df: The full input dataframe
        job_cls: Your concrete SwarmJob class
        job_kwargs: Arguments to instantiate the job (e.g., model, endpoint, etc.)
        tag: Base tag name for checkpoints
        checkpoint_dir: Directory to store per-shard checkpoints
        num_threads: Optional override for how many threads/cores to use

    Returns:
        A single combined DataFrame with all outputs.
    """
    warnings.warn(
        "run_swarm_in_threads is deprecated; sharding handled by run_job_unified.",
        DeprecationWarning,
    )

    def make_job():
        return job_cls(**job_kwargs)

    return await run_job_unified(
        make_job,
        df,
        input_col=job_kwargs.get("input_column_name", "messages"),
        output_cols=[job_kwargs.get("output_column_name", "result")],
        nshards=num_threads or 1,
        store_uri=None,  # disables new-style path
        checkpoint_every=job_kwargs.get("checkpoint_interval", 16),
        tag=tag,
        checkpoint_dir=str(checkpoint_dir),
    )
    MAX_PARALLELISM = os.cpu_count()
    if num_threads is None:
        num_threads = MAX_PARALLELISM if MAX_PARALLELISM is not None else 1

    logger.info(
        f"[bold green]Running job in {num_threads} threads (max: {MAX_PARALLELISM})[/bold green]"
    )

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    shards = np.array_split(df.index, num_threads)
    shards = [df.loc[idx].copy() for idx in shards]

    # TODO: Potentially dangerous
    results: list[pd.DataFrame] = [pd.DataFrame()] * num_threads
    threads: list[threading.Thread] = []

    def thread_worker(i: int, shard_df: pd.DataFrame):
        async def runner():
            job = job_cls(**job_kwargs)
            result = await job.run(
                shard_df, tag=f"{tag}_shard{i}", checkpoint_dir=checkpoint_dir
            )
            results[i] = result

        asyncio.run(runner())

    for i, shard in enumerate(shards):
        t = threading.Thread(target=thread_worker, args=(i, shard), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    logger.info("[bold green]✅ All shards finished. Merging...[/bold green]")

    final_df = pd.concat(results).sort_index()

    # TODO: Find a way to clean up per-shard checkpoints if needed
    # This could be done by removing the individual shard files after merging
    # for i in range(num_threads):
    #     shard_path = checkpoint_dir / f"{job_cls.__class__.__name__}_{tag}_shard{i}.parquet"
    #     if shard_path.exists():
    #         os.remove(shard_path)

    return final_df


def parse_args(cli_args=None):
    if isinstance(cli_args, argparse.Namespace):
        return cli_args

    parser = argparse.ArgumentParser(description="Run a SwarmJob on a Parquet input.")

    parser.add_argument(
        "--job-class", type=str, help="Job class path in format 'pkg.module:Class'"
    )
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--input-parquet", type=Path, help="Path to input Parquet file")
    parser.add_argument(
        "--output-parquet", type=Path, help="Path to output Parquet file"
    )
    parser.add_argument("--endpoint", type=str, help="Endpoint URL")
    parser.add_argument(
        "--job-kwargs", type=str, default="{}", help="Extra JSON string kwargs"
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

    if not cli_args:
        return parser.parse_args()
    return parser.parse_args(args=cli_args)


def build_job_from_args(args) -> tuple[Type[SwarmJob], dict]:
    cls_path = args.job_class or os.environ["JOB_CLASS"]
    model = args.model or os.environ["MODEL"]
    endpoint = args.endpoint or os.environ["ENDPOINT"]
    job_params = json.loads(args.job_kwargs or os.getenv("JOB_KWARGS", "{}"))
    kwargs = job_params.pop("kwargs", {})

    job_cls = _load_cls(cls_path)
    job_kwargs = {"endpoint": endpoint, "model": model, **job_params, **kwargs}
    return job_cls, job_kwargs


async def _amain(cli_args: list[str] | argparse.Namespace | None = None):
    args = parse_args(cli_args)
    in_path: Path = args.input_parquet or Path(os.environ["INPUT_PARQUET"])
    out_path: Path = args.output_parquet or Path(os.environ["OUTPUT_PARQUET"])
    job_cls, job_kwargs = build_job_from_args(args)
    df_in = load_dataframe(in_path, limit=args.limit)
    tag = parquet_hash(in_path) + compute_hash(str(out_path))

    def make_job():
        return job_cls(**job_kwargs)

    # Resolve output columns for new-style (fallback to legacy CLI flag)
    output_cols = None
    if hasattr(job_cls, "output_column_name"):
        o = getattr(job_cls, "output_column_name")
        output_cols = o if isinstance(o, list) else [o]
    elif getattr(args, "output_column", None):
        output_cols = [args.output_column]

    # Map legacy --nthreads to shards
    nshards = getattr(args, "nthreads", 1)

    # Derive a default store URI from checkpoint dir (local file://); keeps legacy alive
    ckp_base = Path(args.checkpoint_dir) / f"{job_cls.__name__}_{tag}.parquet"
    store_uri = f"file://{ckp_base}"

    df_out = await run_job_unified(
        make_job,
        df_in,
        input_col=job_kwargs.get("input_column_name", "messages"),
        output_cols=output_cols,
        nshards=nshards,
        store_uri=store_uri,  # used by new-style
        checkpoint_every=args.checkpoint_interval,
        tag=tag,  # used by old-style
        checkpoint_dir=args.checkpoint_dir,  # used by old-style
    )

    save_dataframe(df_out, out_path)


def main(cli_args: list[str] | None = None):
    args = parse_args(cli_args or sys.argv[1:])
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
