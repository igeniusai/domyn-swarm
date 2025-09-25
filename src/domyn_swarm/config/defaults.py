import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import yaml

from . import get_settings

T = TypeVar("T")
_REQUIRED = object()  # sentinel

# Search order (first hit wins)
_DEFAULT_FILES = (
    # explicit path via env
    lambda: Path(os.environ["DOMYN_SWARM_DEFAULTS"])
    if os.getenv("DOMYN_SWARM_DEFAULTS")
    else None,
    # project local
    lambda: Path.cwd() / "defaults.yaml",
    lambda: Path.cwd() / ".domyn_swarm" / "defaults.yaml",
    # user config dir
    lambda: Path.home() / ".domyn_swarm" / "defaults.yaml",
    # package etc/ (optional; adjust to your tree)
    lambda: Path(__file__).resolve().parents[2] / "etc" / "defaults.yaml",
)


def _find_defaults_file() -> Optional[Path]:
    s = get_settings()
    if s.defaults_file and Path(s.defaults_file).is_file():
        return Path(s.defaults_file)

    for f in _DEFAULT_FILES:
        try:
            p = f()
        except Exception:
            p = None
        if p and p.is_file():
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
    if v is None or (isinstance(v, str) and not v.strip()):
        if fallback is _REQUIRED:
            raise ValueError(f"Missing required default: {key}")
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
