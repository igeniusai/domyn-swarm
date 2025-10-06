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
import os
from pathlib import Path

import pandas as pd

from domyn_swarm import utils


def load_dataframe(path: Path, limit: int | None = None) -> pd.DataFrame:
    match path.suffix.lower():
        case ".parquet":
            df = pd.read_parquet(path)
        case ".csv":
            df = pd.read_csv(path)
        case ".jsonl":
            df = pd.read_json(path, orient="records", lines=True)
        case _:
            raise ValueError(f"Unsupported file format: {path.suffix.lower()}")

    return df.head(limit) if limit else df


def save_dataframe(df: pd.DataFrame, path: Path):
    os.makedirs(path.parent, exist_ok=True)

    match path.suffix.lower():
        case ".parquet":
            df.to_parquet(path, index=False)
        case ".csv":
            df.to_csv(path, index=False)
        case ".jsonl":
            df.to_json(path, orient="records", lines=True)
        case _:
            raise ValueError(f"Unsupported output file format: {path.suffix.lower()}")


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
