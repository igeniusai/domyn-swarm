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

from pathlib import Path

import pandas as pd


class CheckpointManager:
    """Manages checkpointing for job outputs.
    Saves intermediate results to a specified path and allows recovery of
    unfinished jobs by filtering out already processed rows."""

    def __init__(self, path: Path | str, df: pd.DataFrame):
        self.path = Path(path)
        self.df = df
        self.done_df = pd.read_parquet(path) if self.path.exists() else pd.DataFrame()
        self.done_idx = set(self.done_df.index)

    def filter_todo(self) -> pd.DataFrame:
        return self.df.loc[~self.df.index.isin(self.done_idx)].copy()

    def flush(self, out_list, new_ids, output_column_name, idx_map):
        global_indices = [idx_map[i] for i in new_ids]
        tmp = self.df.loc[global_indices].copy()

        if isinstance(output_column_name, str):
            tmp[output_column_name] = [out_list[i] for i in new_ids]
        else:
            for col_idx, col_name in enumerate(output_column_name):
                tmp[col_name] = [out_list[i][col_idx] for i in new_ids]

        self.done_df = pd.concat([self.done_df, tmp])
        self.done_df.to_parquet(self.path)

    def finalize(self) -> pd.DataFrame:
        return self.done_df.sort_index()
