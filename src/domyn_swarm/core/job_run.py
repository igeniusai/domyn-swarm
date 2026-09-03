# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""How a job is run, as one value.

Shared by `DomynLLMSwarm.submit_job` and the CLI, so neither has to spell out
the run parameters itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class JobRunSpec:
    """Everything about how a job runs, apart from the job itself.

    Attributes:
        input_path: Parquet dataset the job reads.
        output_path: Destination the job writes.
        shard_output: Whether shards are written directly from checkpoint
            outputs rather than assembling the merged result in memory first.
            This controls how the result is produced, not the file layout: a
            directory output is written as one parquet file per shard whenever
            `num_shards` > 1 either way. Applies to the pandas and polars
            engines; ignored by the arrow and ray engines. On pandas and polars
            it also makes `global_resume` a no-op, since each shard's own
            checkpoint store already supplies resume.
        checkpoint_dir: Directory holding checkpoint state. `None` leaves
            `submit_job` to default it to the swarm's own checkpoint directory.
        checkpoint_interval: Items processed between checkpoint flushes; `None`
            leaves the job's own setting in place.
        no_resume: Whether to ignore existing checkpoint state.
        no_checkpointing: Whether to disable checkpointing entirely.
        runner: Deprecated runner implementation for non-ray backends,
            consulted only when `engine` is not given.
        engine: Execution engine ("pandas", "arrow", "polars" or "ray"). Takes
            precedence over the deprecated `runner`/`data_backend` pair, and
            defaults to the value resolved from that pair.
        num_shards: Number of shards to split the input into. Part of the
            checkpoint layout, so keep it fixed across resumes of the same job
            or previously-completed rows are reprocessed.
        shard_mode: Sharding strategy for `num_shards` > 1: "id" for stable id
            hashing, "index" for legacy row-order sharding.
        global_resume: Whether to filter inputs using done ids gathered across
            all shards. Ignored on the pandas and polars engines when
            `shard_output` is set.
        limit: Maximum number of input rows to read, for dry runs and
            debugging; `None` processes the whole dataset.
        detach: Whether to start the job in a new process group and return
            immediately rather than blocking until it completes.
        mail_user: Address for scheduler notifications.
        ray_address: Address of an existing Ray cluster.
        checkpoint_tag: Tag distinguishing checkpoint state between runs.
        job_resources: Scheduler resource overrides for the job step.
        num_threads: Deprecated alias for `num_shards`. When set, it overrides
            `num_shards` and emits a `DeprecationWarning`; despite the name,
            the value has always been a shard count, not a thread count.
    """

    input_path: Path
    output_path: Path
    shard_output: bool = False
    checkpoint_dir: Path | None = None
    checkpoint_interval: int | None = None
    no_resume: bool = False
    no_checkpointing: bool = False
    runner: str = "pandas"
    engine: str | None = None
    num_shards: int = 1
    shard_mode: Literal["id", "index"] = "id"
    global_resume: bool = False
    limit: int | None = None
    detach: bool = False
    mail_user: str | None = None
    ray_address: str | None = None
    checkpoint_tag: str | None = None
    job_resources: dict | None = None
    num_threads: int | None = None
