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

from __future__ import annotations

from functools import lru_cache
import importlib

_LEPTON_IMPORT_ERR: Exception | None = None


@lru_cache(maxsize=1)
def _lepton_client_cls():
    """Probe for and return the Lepton SDK ``APIClient`` class lazily (cached).

    Returns the class, or ``None`` if the SDK is unavailable (the last import
    error is recorded in ``_LEPTON_IMPORT_ERR``). The probe runs on first use
    rather than at module import, so importing this module — and anything that
    transitively imports it (config, state, CLI) — does NOT pull in leptonai.
    """
    global _LEPTON_IMPORT_ERR
    # Try modern and fallback module paths (adjust if SDK layout changes)
    for modpath, attr in (
        ("leptonai.api.v2.client", "APIClient"),
        ("leptonai.api.client", "APIClient"),  # fallback for older SDKs
    ):
        try:
            mod = importlib.import_module(modpath)
        except ModuleNotFoundError as e:
            _LEPTON_IMPORT_ERR = e
            continue
        try:
            cls = getattr(mod, attr)
        except Exception as e:
            # Imported module but failed to access symbol (bad install / incompatible deps)
            _LEPTON_IMPORT_ERR = e
            break
        _LEPTON_IMPORT_ERR = None
        return cls
    return None


def have_lepton() -> bool:
    """Return True if the Lepton SDK is importable (client class found)."""
    return _lepton_client_cls() is not None


def _require_lepton() -> None:
    """
    Raise ImportError with a helpful message if the Lepton SDK is unavailable.
    Does NOT instantiate the client; safe to call anywhere.
    """
    if _lepton_client_cls() is None:
        hint = (
            f" (import failed: {type(_LEPTON_IMPORT_ERR).__name__}: {_LEPTON_IMPORT_ERR})"
            if _LEPTON_IMPORT_ERR
            else ""
        )
        raise ImportError(
            "Install `domyn-swarm[lepton]` to use the Lepton backend." + hint
        ) from _LEPTON_IMPORT_ERR


def make_lepton_client(*, token: str | None = None, workspace: str | None = None):
    """
    Instantiate APIClient lazily (no import-time side effects).
    If the SDK requires auth (e.g., `lep login`), pass a token via env/CI.
    """
    _require_lepton()
    client_cls = _lepton_client_cls()
    assert client_cls is not None  # _require_lepton would have raised otherwise
    try:
        if token is not None and workspace is not None:
            return client_cls(token=token, workspace_id=workspace)
        if token is not None:
            return client_cls(token=token)
        if workspace is not None:
            return client_cls(workspace_id=workspace)
        return client_cls()
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize Lepton API client. "
            "Run `lep login` locally or set LEPTONAI_API_TOKEN in CI. "
            "In tests, monkeypatch `make_lepton_client` or your backend's `_client()`."
        ) from e
