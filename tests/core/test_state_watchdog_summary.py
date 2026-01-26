from sqlalchemy.orm import Session

from domyn_swarm.core.state import watchdog as watchdog_state
from domyn_swarm.core.state.db import create_engine_for
from domyn_swarm.core.state.models import Base, ReplicaStatus
from domyn_swarm.runtime.watchdog import ReplicaState


def test_read_swarm_summary_counts(tmp_path):
    """Summarizes replica counts and failure reasons."""
    db_path = tmp_path / "watchdog.db"
    engine = create_engine_for(db_path)
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        session.add_all(
            [
                ReplicaStatus(
                    swarm_id="s1",
                    replica_id=0,
                    state=ReplicaState.RUNNING,
                    http_ready=True,
                ),
                ReplicaStatus(
                    swarm_id="s1",
                    replica_id=1,
                    state=ReplicaState.FAILED,
                    http_ready=False,
                    fail_reason="boom",
                ),
            ]
        )
        session.commit()

    summary = watchdog_state.read_swarm_summary(db_path, swarm_id="s1")
    assert summary is not None
    assert summary.total == 2
    assert summary.running == 1
    assert summary.http_ready == 1
    assert summary.failed == 1
    assert summary.fail_reasons["boom"] == 1


def test_read_swarm_summary_missing_db(tmp_path):
    """Returns None when the database file is missing."""
    db_path = tmp_path / "missing.db"
    assert watchdog_state.read_swarm_summary(db_path, swarm_id="s") is None


def test_list_replica_failures(tmp_path):
    """Returns failure rows with formatted timestamps."""
    db_path = tmp_path / "watchdog.db"
    engine = create_engine_for(db_path)
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        session.add_all(
            [
                ReplicaStatus(
                    swarm_id="s1",
                    replica_id=2,
                    state=ReplicaState.EXITED,
                    http_ready=False,
                    fail_reason="oops",
                ),
                ReplicaStatus(
                    swarm_id="s1",
                    replica_id=3,
                    state=ReplicaState.RUNNING,
                    http_ready=True,
                ),
            ]
        )
        session.commit()

    rows = watchdog_state.list_replica_failures(db_path, "s1")
    assert len(rows) == 1
    assert rows[0]["replica_id"] == 2
    assert rows[0]["fail_reason"] == "oops"
