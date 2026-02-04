from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object


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

    ids_index = ids if isinstance(ids, pd.Index) else pd.Index(ids)
    hashed = hash_pandas_object(pd.Series(ids_index, copy=False), index=False)  # type: ignore[arg-type]
    hashed_arr = np.asarray(hashed, dtype=np.uint64)
    shard_ids = (hashed_arr % np.uint64(nshards)).astype(np.int64, copy=False)
    return [np.flatnonzero(shard_ids == i) for i in range(nshards)]
