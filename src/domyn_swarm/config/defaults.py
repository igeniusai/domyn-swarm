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
from functools import lru_cache
import logging
import os
from pathlib import Path
from typing import Any, TypeVar

import yaml

from domyn_swarm.helpers.logger import setup_logger

from . import get_settings

logger = setup_logger(__name__, level=logging.INFO)

T = TypeVar("T")
_REQUIRED = object()  # sentinel

# Search order (first hit wins)
_DEFAULT_FILES = (
    # explicit path via env
    lambda: Path(os.environ["DOMYN_SWARM_DEFAULTS"]) if os.getenv("DOMYN_SWARM_DEFAULTS") else None,
    # project local
    lambda: Path.cwd() / "defaults.yaml",
    lambda: Path.cwd() / ".domyn_swarm" / "defaults.yaml",
    # user config dir
    lambda: Path.home() / ".domyn_swarm" / "defaults.yaml",
    # package etc/ (optional; adjust to your tree)
    lambda: Path(__file__).resolve().parents[2] / "etc" / "defaults.yaml",
)


def _find_defaults_file() -> Path | None:
    s = get_settings()
    if s.defaults_file and Path(s.defaults_file).is_file():
        return Path(s.defaults_file)

    for f in _DEFAULT_FILES:
        try:
            p = f()
        except Exception:
            p = None
        if p and p.is_file():
            logger.debug(f"Using defaults file: {p}")
            return p
    return None


@lru_cache(maxsize=1)
def _load_defaults() -> dict[str, Any]:
    p = _find_defaults_file()
    if not p:
        return {}
    data = yaml.safe_load(p.read_text()) or {}
    if not isinstance(data, dict):
        return {}
    return data


def _get_by_dots(d: dict[str, Any], key: str) -> Any:
    cur: Any = d
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def get_default(key: str, fallback: T | object = _REQUIRED) -> T:
    """Return defaults.yaml[key]; raise if required and missing/empty."""
    v: Any = _get_by_dots(_load_defaults(), key)
    logger.debug(f"Default for {key!r}: {v!r} (fallback: {fallback!r})")
    if v is None or (isinstance(v, str) and not v.strip()):
        if fallback is _REQUIRED:
            raise ValueError(f"Missing required configuration key: {key}")
        return fallback  # type: ignore[return-value]
    return v  # type: ignore[return-value]


def default_for(key: str, fallback: T | object = _REQUIRED) -> Callable[[], T]:
    """Factory for Field(default_factory=...). Raises if required default is absent/empty."""

    def _factory() -> T:
        return get_default(key, fallback)

    return _factory


def reload_defaults_cache() -> None:
    """If you change defaults.yaml at runtime/tests, clear cache."""
    _load_defaults.cache_clear()
