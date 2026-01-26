import builtins

import domyn_swarm.config.lepton as lepton_mod


def test_default_mounts_falls_back_when_sdk_missing(monkeypatch):
    """Returns dict mounts when the SDK import fails."""
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        """Force leptonai imports to fail."""
        if name.startswith("leptonai"):
            raise ModuleNotFoundError("no lepton")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    mounts = lepton_mod._default_mounts()
    assert isinstance(mounts, list)
    assert isinstance(mounts[0], dict)


def test_coerce_mounts_accepts_none(monkeypatch):
    """Coerces None to an empty list."""
    assert lepton_mod.LeptonEndpointConfig._coerce_mounts(None) == []
