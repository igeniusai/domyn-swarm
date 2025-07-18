import os

import pandas as pd


class CheckpointManager:
    """Manages checkpointing for job outputs.
    Saves intermediate results to a specified path and allows recovery of
    unfinished jobs by filtering out already processed rows."""

    def __init__(self, path: str, df: pd.DataFrame):
        self.path = path
        self.df = df
        self.done_df = pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()
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
