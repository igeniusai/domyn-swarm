# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""The execution skeleton shared by every eager engine.

Everything frame-type-specific lives behind `FrameOps`; this module owns only
the order of operations, which is identical for pandas and Arrow.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, TypeVar

from domyn_swarm.jobs.execution.frame_ops import FrameOps, ShardSpec
from domyn_swarm.jobs.io.checkpointing import (
    _shard_store_uri,
    _validate_checkpoint_store,
    _validate_sharded_execution,
    load_global_done_ids,
)

T = TypeVar("T")


async def run_sharded_pipeline(
    *,
    ops: FrameOps[T],
    job_factory: Callable[[], Any],
    data: Any,
    spec: ShardSpec,
    require_id: bool,
    nshards: int,
    shard_mode: str,
    global_resume: bool,
    store_uri: str | None,
    finalize: Callable[..., T] | None = None,
) -> T:
    """Run a job over a dataset, sharded and resumable.

    Args:
        ops: Adapter supplying the frame-type-specific operations.
        job_factory: Callable producing a fresh job instance per shard.
        data: Backend-native input data.
        spec: Settings shared by every shard.
        require_id: Whether `spec.id_col` must already exist in the input.
        nshards: Number of shards; values <= 1 run the frame unsharded.
        shard_mode: "id" for stable id hashing, "index" for positional order.
        global_resume: Whether to skip ids already recorded across all shards.
        store_uri: Base checkpoint store URI.
        finalize: Called instead of `ops.concat` when resuming globally, to
            join checkpoint outputs back onto the full input.

    Returns:
        The job outputs in the adapter's frame type.

    Raises:
        ValueError: If a required id column is missing, if checkpointing is
            misconfigured, or if `shard_mode` is unsupported.
    """
    frame = ops.coerce(data, spec.id_col)
    if require_id and spec.id_col not in ops.column_names(frame):
        raise ValueError(f"Input is missing required id column {spec.id_col!r}.")

    _validate_checkpoint_store(spec.checkpointing, store_uri)

    if nshards <= 1:
        return await ops.run_shard(job_factory, frame, store_uri, spec)

    _validate_sharded_execution(spec.checkpointing)
    if shard_mode not in {"id", "index"}:
        raise ValueError(f"Unsupported shard_mode: {shard_mode}")

    frame = ops.ensure_id(frame, spec.id_col)
    frame_full = frame

    if global_resume and spec.checkpointing:
        if store_uri is None:
            raise ValueError("store_uri is required when global_resume is enabled.")
        done = load_global_done_ids(
            store_uri=store_uri,
            id_col=spec.id_col,
            nshards=nshards,
            store_factory=ops.store_factory,
            empty_data_factory=lambda: ops.empty_with_id(spec.id_col),
        )
        frame = ops.filter_out_ids(frame, spec.id_col, done)

    indices = ops.shard_indices(frame, spec.id_col, shard_mode, nshards)
    if store_uri is None:
        raise ValueError("store_uri is required when running sharded jobs.")

    take = ops.take if shard_mode == "index" else ops.take_positional

    async def _one(shard_id: int, idx) -> T:
        return await ops.run_shard(
            job_factory, take(frame, idx), _shard_store_uri(store_uri, shard_id), spec
        )

    parts = await asyncio.gather(*[_one(i, idx) for i, idx in enumerate(indices)])

    if global_resume and spec.checkpointing and finalize is not None:
        return finalize(frame_full=frame_full, store_uri=store_uri, nshards=nshards, spec=spec)
    return ops.concat(list(parts))
