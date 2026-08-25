# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3


@dataclass
class ReplicaStatusRow:
    swarm_id: str
    replica_id: int
    node: str | None
    port: int | None
    state: str | None
    http_ready: int | None
    exit_code: int | None
    exit_signal: int | None
    fail_reason: str | None
    last_seen: str | None


def read_replica_statuses(db_path: Path, swarm_id: str) -> list[ReplicaStatusRow]:
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path.as_posix())
    try:
        cur = conn.execute(
            """
            SELECT swarm_id, replica_id, node, port, state, http_ready,
                   exit_code, exit_signal, fail_reason, last_seen
            FROM replica_status
            WHERE swarm_id = ?
            ORDER BY replica_id
            """,
            (swarm_id,),
        )
        rows = cur.fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()

    return [
        ReplicaStatusRow(
            swarm_id=row[0],
            replica_id=row[1],
            node=row[2],
            port=row[3],
            state=row[4],
            http_ready=row[5],
            exit_code=row[6],
            exit_signal=row[7],
            fail_reason=row[8],
            last_seen=row[9],
        )
        for row in rows
    ]
