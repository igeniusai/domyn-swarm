# domyn_swarm/run_job.py  (installed with your library)
import argparse
import logging
import os
import json
import importlib
import pandas as pd
import asyncio
from domyn_swarm.helpers import compute_hash, parquet_hash, setup_logger
from domyn_swarm.jobs import SwarmJob  # base class
from pathlib import Path
import threading
import numpy as np
from typing import Type, Optional, Union

logger = setup_logger("domyn_swarm.run_job", level=logging.INFO)


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

    shards = np.array_split(df, num_threads)
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
    return final_df


def parse_args():
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

    return parser.parse_args()


async def _amain():
    args = parse_args()

    cls_path = args.job_class or os.environ["JOB_CLASS"]
    model = args.model or os.environ["MODEL"]
    in_path = args.input_parquet or Path(os.environ["INPUT_PARQUET"])
    out_path = args.output_parquet or Path(os.environ["OUTPUT_PARQUET"])
    endpoint = args.endpoint or os.environ["ENDPOINT"]

    job_params = json.loads(args.job_kwargs or os.getenv("JOB_KWARGS", "{}"))
    kwargs = job_params.pop("kwargs", {})

    logger.info(f"[bold yellow]Instantiating job {cls_path}")
    JobCls = _load_cls(cls_path)
    job = JobCls(endpoint=endpoint, model=model, **job_params, **kwargs)

    logger.info(f"[bold yellow]Reading input dataset from {in_path}")
    tag = parquet_hash(in_path) + compute_hash(str(out_path))

    # Read the input DataFrame based on the file format
    match in_path.suffix.lower():
        case ".parquet":
            df_in: pd.DataFrame = pd.read_parquet(in_path)
        case ".csv":
            df_in: pd.DataFrame = pd.read_csv(in_path)
        case ".jsonl":
            df_in: pd.DataFrame = pd.read_json(in_path, orient="records", lines=True)
        case _:
            raise ValueError(f"Unsupported file format: {in_path.suffix.lower()}")

    if args.limit:
        df_in = df_in.head(args.limit)

    if args.nthreads <= 1:
        df_out: pd.DataFrame = await job.run(df_in, tag)
    else:
        logger.info(
            f"[bold green]Running job in multithreaded mode (num_threads={args.nthreads})"
        )
        df_out: pd.DataFrame = run_swarm_in_threads(
            df_in,
            JobCls,
            job_kwargs={"endpoint": endpoint, "model": model, **job_params, **kwargs},
            tag=tag,
            num_threads=args.nthreads,
        )

    logger.info(f"[bold green]Saving output dataset to {out_path}")
    os.makedirs(out_path.parent, exist_ok=True)

    # Save the output DataFrame to the specified format
    match out_path.suffix.lower():
        case ".parquet":
            df_out.to_parquet(out_path, index=False)
        case ".csv":
            df_out.to_csv(out_path, index=False)
        case ".jsonl":
            df_out.to_json(out_path, orient="records", lines=True)
        case _:
            raise ValueError(
                f"Unsupported output file format: {out_path.suffix.lower()}"
            )


def main():
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
