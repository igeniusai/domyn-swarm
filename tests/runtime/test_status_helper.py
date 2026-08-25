# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sqlite3

from domyn_swarm.runtime.collector import ensure_schema
from domyn_swarm.runtime.status import read_replica_statuses


def test_read_replica_statuses(tmp_path: Path):
    db_path = tmp_path / "watchdog.db"
    conn = sqlite3.connect(db_path.as_posix())
    ensure_schema(conn)

    conn.execute(
        """
        INSERT INTO replica_status
        (swarm_id, replica_id, node, port, state, http_ready, exit_code, exit_signal, fail_reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("s1", 0, "node0", 9000, "running", 1, None, None, None),
    )
    conn.commit()
    conn.close()

    rows = read_replica_statuses(db_path, "s1")
    assert len(rows) == 1
    assert rows[0].replica_id == 0
    assert rows[0].state == "running"
