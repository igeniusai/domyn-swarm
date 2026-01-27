from pathlib import Path

from domyn_swarm.cli import db as db_mod
from domyn_swarm.core import swarm as swarm_mod
from domyn_swarm.platform.protocols import ServingPhase, ServingStatus


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


def _fake_swarm_with_status(status: ServingStatus):
    class _Serving:
        def status(self, handle):
            return status

    class _Dep:
        serving = _Serving()

    class _Swarm:
        _deployment = _Dep()
        serving_handle = object()

    return _Swarm()


def test_db_prune_deletes_dirty_and_failed_load(monkeypatch, capsys):
    records = [
        {"deployment_name": "clean"},
        {"deployment_name": "boom"},
        {"deployment_name": "failed"},
    ]
    monkeypatch.setattr(db_mod.SwarmStateManager, "list_all", lambda: records)

    def _from_state(*, deployment_name: str):
        if deployment_name == "clean":
            return _fake_swarm_with_status(
                ServingStatus(phase=ServingPhase.RUNNING, url=None, detail=None)
            )
        if deployment_name == "failed":
            return _fake_swarm_with_status(
                ServingStatus(phase=ServingPhase.FAILED, url=None, detail=None)
            )
        raise RuntimeError("boom")

    monkeypatch.setattr(swarm_mod.DomynLLMSwarm, "from_state", staticmethod(_from_state))

    deleted = {}

    def _delete_records(names):
        deleted["names"] = names
        return len(names)

    monkeypatch.setattr(db_mod.SwarmStateManager, "delete_records", _delete_records)

    db_mod.db_prune(yes=True)
    assert deleted["names"] == ["boom", "failed"]
    assert "Deleted 2 swarm record(s)." in capsys.readouterr().out


def test_db_prune_no_dirty(monkeypatch, capsys):
    records = [{"deployment_name": "clean"}]
    monkeypatch.setattr(db_mod.SwarmStateManager, "list_all", lambda: records)

    def _from_state(*, deployment_name: str):
        return _fake_swarm_with_status(
            ServingStatus(phase=ServingPhase.RUNNING, url=None, detail=None)
        )

    monkeypatch.setattr(swarm_mod.DomynLLMSwarm, "from_state", staticmethod(_from_state))
    monkeypatch.setattr(
        db_mod.SwarmStateManager,
        "delete_records",
        lambda names: (_ for _ in ()).throw(AssertionError("delete_records called")),
    )

    db_mod.db_prune(yes=True)
    assert "No dirty swarm records found." in capsys.readouterr().out
