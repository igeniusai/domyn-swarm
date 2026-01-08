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

import os
from pathlib import Path

import pandas as pd

from domyn_swarm.helpers.logger import setup_logger

logger = setup_logger(__name__)


class CheckpointManager:
    """Manages checkpointing for job outputs.
    Saves intermediate results to a specified path and allows recovery of
    unfinished jobs by filtering out already processed rows."""

    def __init__(
        self,
        path: Path | str,
        df: pd.DataFrame,
        *,
        expected_output_cols: str | list[str] | None = None,
    ):
        self.path = Path(path)
        self.df = df
        self.done_df = pd.read_parquet(path) if self.path.exists() else pd.DataFrame()
        if not self.done_df.empty:
            if not self.done_df.index.is_unique:
                raise ValueError("Checkpoint index contains duplicate rows.")
            if expected_output_cols is not None:
                expected = (
                    [expected_output_cols]
                    if isinstance(expected_output_cols, str)
                    else list(expected_output_cols)
                )
                missing = [c for c in expected if c not in self.done_df.columns]
                if missing:
                    raise ValueError(
                        f"Checkpoint is missing expected output columns: {', '.join(missing)}"
                    )
            stray = self.done_df.index.difference(self.df.index)
            if len(stray) > 0:
                logger.warning(
                    "Checkpoint contains %d rows not present in input; ignoring them.",
                    len(stray),
                )
                self.done_df = self.done_df.loc[self.done_df.index.intersection(self.df.index)]
        self.done_idx = set(self.done_df.index)

    def filter_todo(self) -> pd.DataFrame:
        return self.df.loc[~self.df.index.isin(self.done_idx)].copy()

    def flush(self, out_list, new_ids, output_cols, idx_map):
        global_indices = [idx_map[i] for i in new_ids]
        tmp = self.df.loc[global_indices].copy()

        if isinstance(output_cols, str):
            tmp[output_cols] = [out_list[i] for i in new_ids]
        else:
            for col_idx, col_name in enumerate(output_cols):
                tmp[col_name] = [out_list[i][col_idx] for i in new_ids]

        self.done_df = pd.concat([self.done_df, tmp])
        self._atomic_write(self.done_df, self.path)

    def finalize(self) -> pd.DataFrame:
        return self.done_df.sort_index()

    @staticmethod
    def _atomic_write(df: pd.DataFrame, path: Path) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        df.to_parquet(tmp_path)
        os.replace(tmp_path, path)
