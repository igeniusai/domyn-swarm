from __future__ import annotations

import importlib
from typing import Optional

_LeptonAPIClient = None  # type: ignore[assignment]
_LEPTON_IMPORT_ERR: Exception | None = None

# Try modern and fallback module paths (adjust if SDK layout changes)
for modpath, attr in (
    ("leptonai.api.v2.client", "APIClient"),
    ("leptonai.api.client", "APIClient"),  # fallback for older SDKs
):
    try:
        mod = importlib.import_module(modpath)
        _LeptonAPIClient = getattr(mod, attr)
        _LEPTON_IMPORT_ERR = None
        break
    except ModuleNotFoundError as e:
        _LEPTON_IMPORT_ERR = e
        continue
    except Exception as e:
        # Imported module but failed to access symbol (bad install / incompatible deps)
        _LEPTON_IMPORT_ERR = e
        break


def have_lepton() -> bool:
    """Return True if the Lepton SDK is importable (client class found)."""
    return _LeptonAPIClient is not None


def _require_lepton() -> None:
    """
    Raise ImportError with a helpful message if the Lepton SDK is unavailable.
    Does NOT instantiate the client; safe to call anywhere.
    """
    if _LeptonAPIClient is None:
        hint = (
            f" (import failed: {type(_LEPTON_IMPORT_ERR).__name__}: {_LEPTON_IMPORT_ERR})"
            if _LEPTON_IMPORT_ERR
            else ""
        )
        raise ImportError(
            "Install `domyn-swarm[lepton]` to use the Lepton backend." + hint
        ) from _LEPTON_IMPORT_ERR


def make_lepton_client(*, token: Optional[str] = None, workspace: Optional[str] = None):
    """
    Instantiate APIClient lazily (no import-time side effects).
    If the SDK requires auth (e.g., `lep login`), pass a token via env/CI.
    """
    _require_lepton()
    try:
        if token is not None and workspace is not None:
            return _LeptonAPIClient(token=token, workspace_id=workspace)  # type: ignore[misc]
        if token is not None:
            return _LeptonAPIClient(token=token)  # type: ignore[misc]
        if workspace is not None:
            return _LeptonAPIClient(workspace_id=workspace)  # type: ignore[misc]
        return _LeptonAPIClient()  # type: ignore[misc]
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize Lepton API client. "
            "Run `lep login` locally or set LEPTONAI_API_TOKEN in CI. "
            "In tests, monkeypatch `make_lepton_client` or your backend’s `_client()`."
        ) from e
