from pathlib import Path

from domyn_swarm.cli import db as db_mod


class DummySettings:
    """Settings stub with a home directory."""

    def __init__(self, home: Path):
        self.home = home


def test_db_upgrade_uses_home(monkeypatch, tmp_path):
    """Invokes upgrade_head with the swarm.db path."""
    called = {}
    monkeypatch.setattr(db_mod, "get_settings", lambda: DummySettings(tmp_path))
    monkeypatch.setattr(db_mod, "upgrade_head", lambda path: called.setdefault("path", path))
    db_mod.db_upgrade()
    assert called["path"].endswith("swarm.db")


def test_db_stamp_uses_home(monkeypatch, tmp_path):
    """Invokes stamp_head with the swarm.db path."""
    called = {}
    monkeypatch.setattr(db_mod, "get_settings", lambda: DummySettings(tmp_path))
    monkeypatch.setattr(db_mod, "stamp_head", lambda path: called.setdefault("path", path))
    db_mod.db_stamp()
    assert called["path"].endswith("swarm.db")
