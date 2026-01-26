from domyn_swarm.runtime import collector as collector_mod


def test_normalize_payload_validates_required_fields():
    """Returns None for payloads missing required fields."""
    assert collector_mod._normalize_payload({}) is None
    assert collector_mod._normalize_payload({"swarm_id": "s"}) is None
    assert collector_mod._normalize_payload({"swarm_id": "s", "replica_id": "bad"}) is None


def test_upsert_status_persists_row(tmp_path):
    """Upserts a normalized payload into the SQLite database."""
    db_path = tmp_path / "watchdog.db"
    conn = collector_mod.open_db(db_path)
    payload = {
        "swarm_id": "swarm-1",
        "replica_id": 0,
        "node": "node-a",
        "port": "8000",
        "pid": 123,
        "state": "running",
        "http_ready": True,
        "exit_code": None,
        "exit_signal": None,
        "fail_reason": None,
        "agent_version": "v1",
    }
    collector_mod.upsert_status(conn, payload)

    row = conn.execute(
        f"SELECT swarm_id, replica_id, node, port, pid, state, http_ready "
        f"FROM {collector_mod.REPLICA_STATUS_TABLE}"
    ).fetchone()
    assert row == ("swarm-1", 0, "node-a", 8000, 123, "running", 1)
