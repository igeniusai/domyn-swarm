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

# helpers/io.py
import glob
import os
from pathlib import Path

import pandas as pd

from domyn_swarm import utils


def load_dataframe(path: Path, limit: int | None = None) -> pd.DataFrame:
    """
    Load a dataframe from various file formats.

    Args:
        path (Path): Path to the file or directory to load data from.
        limit (int | None, optional):  . Defaults to None.

    Raises:
        ValueError:
            - If no files are found in the specified directory.
            - If no files match the provided glob pattern.
            - If mixed file extensions are detected.
            - If an unsupported file format is encountered.
    Returns:
        pd.DataFrame: Loaded dataframe.
    """

    def _read_single(file_path: Path, suffix: str) -> pd.DataFrame:
        match suffix:
            case ".parquet":
                return pd.read_parquet(file_path)
            case ".csv":
                return pd.read_csv(file_path)
            case ".jsonl":
                return pd.read_json(file_path, orient="records", lines=True)
            case _:
                raise ValueError(f"Unsupported file format: {suffix}")

    path = Path(path)

    path_str = str(path)
    has_wildcard = any(ch in path_str for ch in ("*", "?", "["))

    if path.is_dir():
        # Pandas supports reading a directory of parquet files directly.
        df = pd.read_parquet(path_str)
        return df.head(limit) if limit else df

    if has_wildcard:
        matched = sorted(glob.glob(path_str))
        if not matched:
            raise ValueError(f"No files matched glob pattern: {path}")
        suffixes = {Path(p).suffix.lower() for p in matched}
        if suffixes != {".parquet"}:
            raise ValueError(f"Unsupported file format for glob pattern: {suffixes}")
        df = pd.concat((_read_single(Path(p), ".parquet") for p in matched), ignore_index=True)
        return df.head(limit) if limit else df

    df = _read_single(path, path.suffix.lower())

    return df.head(limit) if limit else df


def save_dataframe(df: pd.DataFrame, path: Path):
    path = Path(path)
    is_dir_target = path.is_dir()
    has_suffix = path.suffix != ""

    if is_dir_target or not has_suffix:
        # Allow parquet datasets to be written to a directory (or suffix-less path).
        dest_dir = path
        os.makedirs(dest_dir, exist_ok=True)
        target = dest_dir / "part-0.parquet"
        suffix = ".parquet"
    else:
        os.makedirs(path.parent, exist_ok=True)
        target = path
        suffix = path.suffix.lower()

    match suffix:
        case ".parquet":
            df.to_parquet(target, index=False)
        case ".csv":
            df.to_csv(target, index=False)
        case ".jsonl":
            df.to_json(target, orient="records", lines=True)
        case _:
            raise ValueError(f"Unsupported output file format: {suffix}")


def to_path(path: Path | str) -> Path:
    """
    Return the path given a string
    """
    if isinstance(path, str):
        return utils.EnvPath(path)
    return path


def is_folder(path: str):
    return utils.EnvPath(path).is_dir()


def path_exists(path: str):
    return os.path.exists(path)
