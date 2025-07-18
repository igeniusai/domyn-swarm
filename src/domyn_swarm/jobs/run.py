# domyn_swarm/jobs/run.py
import argparse
import asyncio
import importlib
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Optional, Type, Union

import numpy as np
import pandas as pd

from domyn_swarm.helpers.data import compute_hash, parquet_hash
from domyn_swarm.helpers.io import load_dataframe, save_dataframe
from domyn_swarm.helpers.logger import setup_logger
from domyn_swarm.jobs import SwarmJob  # base class

logger = setup_logger("domyn_swarm.jobs.run", level=logging.INFO)


def _load_cls(path: str) -> type[SwarmJob]:
    mod, cls = path.split(":")
    return getattr(importlib.import_module(mod), cls)


def run_swarm_in_threads(
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

    MAX_PARALLELISM = os.cpu_count()
    num_threads = num_threads or min(MAX_PARALLELISM, max(1, os.cpu_count() or 1))

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    shards = np.array_split(df.index, num_threads)
    shards = [df.loc[idx].copy() for idx in shards]

    results = [None] * num_threads
    threads: list[threading.Thread] = []

    def thread_worker(i: int, shard_df: pd.DataFrame):
        async def runner():
            job = job_cls(**job_kwargs)
            result = await job.run(
                shard_df, tag=f"{tag}_shard{i}", checkpoint_dir=str(checkpoint_dir)
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

    return parser.parse_args(cli_args)


def build_job_from_args(args) -> tuple[Type[SwarmJob], dict]:
    cls_path = args.job_class or os.environ["JOB_CLASS"]
    model = args.model or os.environ["MODEL"]
    endpoint = args.endpoint or os.environ["ENDPOINT"]
    job_params = json.loads(args.job_kwargs or os.getenv("JOB_KWARGS", "{}"))
    kwargs = job_params.pop("kwargs", {})

    job_cls = _load_cls(cls_path)
    job_kwargs = {"endpoint": endpoint, "model": model, **job_params, **kwargs}
    return job_cls, job_kwargs


async def _amain(cli_args: list[str] | None = None):
    args = parse_args(cli_args if cli_args is not None else sys.argv[1:])

    in_path = args.input_parquet or Path(os.environ["INPUT_PARQUET"])
    out_path = args.output_parquet or Path(os.environ["OUTPUT_PARQUET"])

    job_cls, job_kwargs = build_job_from_args(args)
    logger.info(
        f"[bold yellow]Instantiating job {job_cls.__name__} with params: {job_kwargs}"
    )
    job = job_cls(**job_kwargs)

    logger.info(f"[bold yellow]Reading input dataset from {in_path}")
    tag = parquet_hash(in_path) + compute_hash(str(out_path))

    df_in = load_dataframe(in_path, limit=args.limit)

    if args.nthreads <= 1:
        df_out: pd.DataFrame = await job.run(df_in, tag)
    else:
        logger.info(
            f"[bold green]Running job in multithreaded mode (num_threads={args.nthreads})"
        )
        df_out: pd.DataFrame = run_swarm_in_threads(
            df_in,
            job_cls,
            job_kwargs=job_kwargs,
            tag=tag,
            num_threads=args.nthreads,
        )

    logger.info(f"[bold green]Saving output dataset to {out_path}")

    save_dataframe(df_out, out_path)


def main(cli_args: list[str] | None = None):
    args = parse_args(cli_args or sys.argv[1:])
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
