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

import json
import os
from pathlib import Path
import time
from typing import cast
import uuid

from blake3 import blake3
import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object

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
        input_col: str | None = None,
        lock_timeout_s: float = 30.0,
        lock_poll_interval_s: float = 0.1,
    ):
        self.path = Path(path)
        self.df = df
        self.parts_dir = self.path.with_suffix(self.path.suffix + ".parts")
        self.parts_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.path.with_suffix(self.path.suffix + ".meta.json")
        self.input_col = input_col
        self.lock_timeout_s = lock_timeout_s
        self.lock_poll_interval_s = lock_poll_interval_s
        self.done_df = pd.read_parquet(path) if self.path.exists() else pd.DataFrame()
        self._existing_parts = self._list_parts()
        self._new_parts: list[Path] = []
        self._ensure_meta()
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
        self._load_part_indexes()

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

        part_path = self.parts_dir / f"part-{uuid.uuid4().hex}.parquet"
        with self._file_lock():
            self._atomic_write(tmp, part_path)
        self._new_parts.append(part_path)
        self.done_idx.update(tmp.index)

    def finalize(self) -> pd.DataFrame:
        parts = self._existing_parts + self._new_parts
        if parts:
            with self._file_lock():
                merged = self._merge_parts(parts)
                self._atomic_write(merged, self.path)
                for part in parts:
                    part.unlink(missing_ok=True)
                self.done_df = merged
                self._existing_parts = []
                self._new_parts = []
        return self.done_df.sort_index()

    @staticmethod
    def _atomic_write(df: pd.DataFrame, path: Path) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        df.to_parquet(tmp_path)
        os.replace(tmp_path, path)

    def _merge_parts(self, parts: list[Path]) -> pd.DataFrame:
        dfs = [self.done_df] if not self.done_df.empty else []
        dfs.extend(pd.read_parquet(p) for p in parts)
        if not dfs:
            return pd.DataFrame().set_index(self.df.index.name)
        out = pd.concat(dfs)
        out = out.loc[~out.index.duplicated(keep="last")].sort_index()
        return out

    def _list_parts(self) -> list[Path]:
        if not self.parts_dir.exists():
            return []
        return sorted(self.parts_dir.glob("part-*.parquet"))

    def _load_part_indexes(self) -> None:
        for part in self._existing_parts:
            part_df = pd.read_parquet(part)
            self.done_idx.update(part_df.index)

    def _ensure_meta(self) -> None:
        if self.input_col is None:
            return
        fingerprint = self._compute_fingerprint(self.df, self.input_col)
        if self.meta_path.exists():
            meta = json.loads(self.meta_path.read_text())
            if meta.get("fingerprint") != fingerprint or meta.get("input_col") != self.input_col:
                raise ValueError("Checkpoint input fingerprint does not match current data.")
            return
        meta = {"fingerprint": fingerprint, "input_col": self.input_col}
        tmp_path = self.meta_path.with_suffix(self.meta_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(meta))
        os.replace(tmp_path, self.meta_path)

    @staticmethod
    def _compute_fingerprint(df: pd.DataFrame, input_col: str) -> str:
        hasher = blake3()
        index_series = pd.Series(df.index, dtype="object")
        index_hash = hash_pandas_object(index_series)  # type: ignore[arg-type]
        hasher.update(np.asarray(index_hash, dtype="uint64").tobytes())
        if input_col in df.columns:
            col_series = cast(pd.Series, df[input_col])
            col_hash = hash_pandas_object(col_series)  # type: ignore[arg-type]
            hasher.update(np.asarray(col_hash, dtype="uint64").tobytes())
        return hasher.hexdigest()

    def _lock_path(self) -> Path:
        return self.path.with_suffix(self.path.suffix + ".lock")

    def _file_lock(self):
        lock_path = self._lock_path()
        deadline = time.monotonic() + self.lock_timeout_s

        class _Lock:
            def __init__(self, outer):
                self.outer = outer
                self.fd: int | None = None

            def __enter__(self):
                while True:
                    self.fd = self._try_acquire(lock_path)
                    if self.fd is not None:
                        os.write(self.fd, str(os.getpid()).encode("ascii"))
                        return self
                    if time.monotonic() > deadline:
                        raise RuntimeError("Timed out waiting for checkpoint lock.")
                    self._cleanup_stale(lock_path, self.outer.lock_timeout_s)
                    time.sleep(self.outer.lock_poll_interval_s)

            def __exit__(self, exc_type, exc, tb):
                if self.fd is not None:
                    os.close(self.fd)
                lock_path.unlink(missing_ok=True)

            @staticmethod
            def _try_acquire(path: Path) -> int | None:
                try:
                    return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except FileExistsError:
                    return None

            @staticmethod
            def _cleanup_stale(path: Path, timeout_s: float) -> None:
                try:
                    if time.time() - path.stat().st_mtime > timeout_s:
                        path.unlink(missing_ok=True)
                except FileNotFoundError:
                    return

        return _Lock(self)
