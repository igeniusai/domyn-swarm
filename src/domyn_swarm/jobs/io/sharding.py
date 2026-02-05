from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
from pandas.util import hash_array


def shard_indices_by_id(
    ids: pd.Series | pd.Index | Sequence[Any], nshards: int
) -> list[np.ndarray]:
    """Return row indices per shard using a stable hash of ids.

    Args:
        ids: Sequence of stable row ids.
        nshards: Number of shards to split into.

    Returns:
        List of numpy index arrays (one per shard), using positional indices.
    """
    if nshards <= 1:
        return [np.arange(len(ids))]

    arr = np.asarray(ids)
    _hash_array = cast(Callable[[np.ndarray], np.ndarray], hash_array)
    hashed_arr = _hash_array(arr)
    shard_ids = (hashed_arr % np.uint64(nshards)).astype(np.int64, copy=False)
    return [np.flatnonzero(shard_ids == i) for i in range(nshards)]
