import pytest

import domyn_swarm.utils.imports as imports_mod


def test_have_lepton_false_when_missing(monkeypatch):
    """Reports missing Lepton SDK when the client class is unavailable.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setattr(imports_mod, "_LeptonAPIClient", None)
    assert imports_mod.have_lepton() is False


def test_require_lepton_raises_import_error(monkeypatch):
    """Raises ImportError with guidance when Lepton SDK is missing.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setattr(imports_mod, "_LeptonAPIClient", None)
    with pytest.raises(ImportError, match="domyn-swarm\\[lepton\\]"):
        imports_mod._require_lepton()


def test_make_lepton_client_uses_overrides(monkeypatch):
    """Passes token/workspace into the API client constructor.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """

    calls: list[dict] = []

    class DummyClient:
        """Fake Lepton client that records init kwargs."""

        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(imports_mod, "_LeptonAPIClient", DummyClient)
    monkeypatch.setattr(imports_mod, "_LEPTON_IMPORT_ERR", None)

    client = imports_mod.make_lepton_client(token="tok", workspace="ws")
    assert isinstance(client, DummyClient)
    assert calls == [{"token": "tok", "workspace_id": "ws"}]


def test_make_lepton_client_wraps_errors(monkeypatch):
    """Wraps SDK initialization errors with a helpful RuntimeError.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """

    class BoomClient:
        """Fake Lepton client that always fails."""

        def __init__(self, **kwargs):
            raise ValueError("boom")

    monkeypatch.setattr(imports_mod, "_LeptonAPIClient", BoomClient)
    monkeypatch.setattr(imports_mod, "_LEPTON_IMPORT_ERR", None)

    with pytest.raises(RuntimeError, match="Failed to initialize Lepton API client"):
        imports_mod.make_lepton_client()
