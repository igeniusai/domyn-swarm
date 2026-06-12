import domyn_swarm.core.state.autoupgrade as autoupgrade_mod


class DummySettings:
    """Settings stub for autoupgrade tests."""

    def __init__(self, skip: bool):
        self.skip_db_upgrade = skip


def test_ensure_db_up_to_date_skips_when_env(monkeypatch):
    """Returns early when skip flag is set."""
    monkeypatch.setattr(autoupgrade_mod, "_DB_UPGRADED", False)
    monkeypatch.setattr(autoupgrade_mod, "get_settings", lambda: DummySettings(skip=True))
    autoupgrade_mod.ensure_db_up_to_date(noisy=False)
    assert autoupgrade_mod._DB_UPGRADED is True


def test_ensure_db_up_to_date_creates_db(monkeypatch, tmp_path):
    """Calls upgrade_head when DB file is missing."""
    db_path = tmp_path / "swarm.db"
    monkeypatch.setattr(autoupgrade_mod, "_DB_UPGRADED", False)
    monkeypatch.setattr(autoupgrade_mod, "get_settings", lambda: DummySettings(skip=False))
    monkeypatch.setattr(autoupgrade_mod.SwarmStateManager, "_resolve_db_path", lambda: db_path)

    called = {}

    def _upgrade_head(path: str):
        """Record the upgraded DB path."""
        called["path"] = path

    monkeypatch.setattr(autoupgrade_mod, "upgrade_head", _upgrade_head)
    autoupgrade_mod.ensure_db_up_to_date(noisy=False)
    assert called["path"] == db_path.as_posix()


def test_ensure_db_up_to_date_noop_when_current(monkeypatch, tmp_path):
    """No-ops when current revision matches head."""
    db_path = tmp_path / "swarm.db"
    db_path.write_text("x")
    monkeypatch.setattr(autoupgrade_mod, "_DB_UPGRADED", False)
    monkeypatch.setattr(autoupgrade_mod, "get_settings", lambda: DummySettings(skip=False))
    monkeypatch.setattr(autoupgrade_mod.SwarmStateManager, "_resolve_db_path", lambda: db_path)
    monkeypatch.setattr(autoupgrade_mod, "get_current_rev", lambda _: "abc")
    monkeypatch.setattr(autoupgrade_mod, "get_head_rev", lambda _: "abc")

    called = {"count": 0}

    def _upgrade_head(_: str):
        """Track unexpected upgrade attempts."""
        called["count"] += 1

    monkeypatch.setattr(autoupgrade_mod, "upgrade_head", _upgrade_head)
    autoupgrade_mod.ensure_db_up_to_date(noisy=False)
    assert called["count"] == 0


def test_ensure_db_up_to_date_fast_path_skips_alembic(monkeypatch, tmp_path):
    """When the DB is already at head, the cheap fast path returns without ever
    calling the alembic-backed get_current_rev/get_head_rev/upgrade_head."""
    import sqlite3

    from domyn_swarm.core.state.migrate import head_rev_fast

    head = head_rev_fast()
    assert head is not None

    db_path = tmp_path / "swarm.db"
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE alembic_version (version_num VARCHAR(32))")
    con.execute("INSERT INTO alembic_version VALUES (?)", (head,))
    con.commit()
    con.close()

    monkeypatch.setattr(autoupgrade_mod, "_DB_UPGRADED", False)
    monkeypatch.setattr(autoupgrade_mod, "get_settings", lambda: DummySettings(skip=False))
    monkeypatch.setattr(autoupgrade_mod.SwarmStateManager, "_resolve_db_path", lambda: db_path)

    def _boom(*_a, **_k):
        raise AssertionError("alembic-backed path should not be reached on the fast path")

    monkeypatch.setattr(autoupgrade_mod, "get_current_rev", _boom)
    monkeypatch.setattr(autoupgrade_mod, "get_head_rev", _boom)
    monkeypatch.setattr(autoupgrade_mod, "upgrade_head", _boom)

    autoupgrade_mod.ensure_db_up_to_date(noisy=False)
    assert autoupgrade_mod._DB_UPGRADED is True
