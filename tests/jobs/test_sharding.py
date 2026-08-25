# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd

from domyn_swarm.jobs.io.sharding import shard_indices_by_id


def test_shard_indices_by_id_stable_and_complete():
    ids = [10, 11, 12, 13, 14, 15, 16]
    nshards = 3

    idxs_list = shard_indices_by_id(ids, nshards)
    idxs_series = shard_indices_by_id(pd.Series(ids), nshards)
    idxs_index = shard_indices_by_id(pd.Index(ids), nshards)

    flat = np.sort(np.concatenate(idxs_list)).tolist()
    assert flat == list(range(len(ids)))

    as_list = [arr.tolist() for arr in idxs_list]
    assert as_list == [arr.tolist() for arr in idxs_series]
    assert as_list == [arr.tolist() for arr in idxs_index]


def test_shard_indices_by_id_single_shard():
    ids = [1, 2, 3]
    idxs = shard_indices_by_id(ids, 1)
    assert len(idxs) == 1
    assert idxs[0].tolist() == [0, 1, 2]
