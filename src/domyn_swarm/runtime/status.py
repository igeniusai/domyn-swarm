# Copyright 2025 iGenius S.p.A
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
