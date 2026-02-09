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

from collections.abc import Callable
from typing import Any

from domyn_swarm.checkpoint.store import InMemoryStore, ParquetShardStore


def _require_store_uri(store_uri: str | None) -> str:
    """Return the store URI or raise if missing.

    Args:
        store_uri: Store URI for checkpointing.

    Returns:
        Store URI string.

    Raises:
        ValueError: If store_uri is None.
    """
    if store_uri is None:
        raise ValueError("store_uri is required when checkpointing is enabled.")
    return store_uri


def _validate_checkpoint_store(checkpointing: bool, store_uri: str | None) -> None:
    """Validate checkpointing prerequisites.

    Args:
        checkpointing: Whether checkpointing is enabled.
        store_uri: Store URI for checkpointing.

    Raises:
        ValueError: If checkpointing is enabled without a store URI.
    """
    if checkpointing:
        _require_store_uri(store_uri)


def _validate_sharded_execution(checkpointing: bool) -> None:
    """Validate sharded execution prerequisites.

    Args:
        checkpointing: Whether checkpointing is enabled.

    Raises:
        ValueError: If sharded execution is attempted without checkpointing.
    """
    if not checkpointing:
        raise ValueError("Sharded execution requires checkpointing to be enabled.")


def _shard_store_uri(store_uri: str, shard_id: int) -> str:
    """Return the store URI for a specific shard.

    Args:
        store_uri: Base checkpoint store URI.
        shard_id: Zero-based shard index.

    Returns:
        Store URI for the shard.
    """
    return store_uri.replace(".parquet", f"_shard{shard_id}.parquet")


def _shard_filename(shard_id: int, nshards: int) -> str:
    """Return a shard filename for directory outputs.

    Args:
        shard_id: Zero-based shard index.
        nshards: Total number of shards.

    Returns:
        Filename for the shard (e.g., `data-00.parquet`).
    """
    width = max(1, len(str(nshards - 1)))
    return f"data-{shard_id:0{width}d}.parquet"


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
        store_uri = _require_store_uri(store_uri)
        return ParquetShardStore(store_uri)
    return InMemoryStore()


def load_global_done_ids(
    *,
    store_uri: str,
    id_col: str,
    nshards: int,
    store_factory: Callable[[str], Any],
    empty_data_factory: Callable[[], Any],
) -> set[Any]:
    """Collect done ids across all shard checkpoint stores.

    Args:
        store_uri: Base checkpoint store URI.
        id_col: Column name for row ids.
        nshards: Number of shards to scan.
        store_factory: Callable that returns a checkpoint store for a shard URI.
        empty_data_factory: Callable that returns an empty dataset for store.prepare().

    Returns:
        Set of ids already present in checkpoint outputs across all shards.
    """
    done_ids: set[Any] = set()
    for shard_id in range(nshards):
        shard_uri = _shard_store_uri(store_uri, shard_id)
        store = store_factory(shard_uri)
        _ = store.prepare(empty_data_factory(), id_col)
        done_ids.update(getattr(store, "done_ids", set()))
    return done_ids
