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

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.exc import SQLAlchemyError

from domyn_swarm.core.state.db import make_session_factory
from domyn_swarm.core.state.models import ReplicaStatus
from domyn_swarm.runtime.watchdog import ReplicaState


@dataclass
class SwarmReplicaSummary:
    total: int
    running: int
    http_ready: int
    failed: int

    fail_reasons: dict[str, int]
    example_fail_reason: str | None = None


def read_swarm_summary(db_path: Path, swarm_id: str) -> SwarmReplicaSummary | None:
    """
    Read a coarse-grained summary for a swarm from the watchdog DB via SQLAlchemy.

    Returns None if:
      - the DB file does not exist
      - the replica_status table is missing
      - there are no rows yet for this swarm_id
      - or any SQLAlchemy error occurs.
    """
    if not db_path.exists():
        return None

    SessionLocal = make_session_factory(db_path)

    try:
        with SessionLocal() as session:
            rows = (
                session.query(
                    ReplicaStatus.state, ReplicaStatus.http_ready, ReplicaStatus.fail_reason
                )
                .filter(ReplicaStatus.swarm_id == swarm_id)
                .all()
            )
    except SQLAlchemyError:
        # Missing table, incompatible schema, etc. → treat as "no info yet"
        return None

    if not rows:
        return None

    total = len(rows)
    running = 0
    http_ready = 0
    failed = 0

    reason_counter: Counter[str] = Counter()

    for state, ready, fail_reason in rows:
        if state == ReplicaState.RUNNING:
            running += 1
        if bool(ready):
            http_ready += 1
        if state in (ReplicaState.FAILED, ReplicaState.EXITED):
            failed += 1
            # Normalize the reason a bit, but keep it simple
            if fail_reason:
                reason_counter[fail_reason] += 1

    # Pick a representative reason if we have any
    example_reason: str | None = None
    if reason_counter:
        example_reason = reason_counter.most_common(1)[0][0]

    return SwarmReplicaSummary(
        total=total,
        running=running,
        http_ready=http_ready,
        failed=failed,
        fail_reasons=dict(reason_counter),
        example_fail_reason=example_reason,
    )


def list_replica_failures(db_path: Path, swarm_name: str) -> list[dict]:
    SessionLocal = make_session_factory(db_path)

    with SessionLocal() as session:
        rows = (
            session.query(
                ReplicaStatus.replica_id,
                ReplicaStatus.node,
                ReplicaStatus.port,
                ReplicaStatus.state,
                ReplicaStatus.exit_code,
                ReplicaStatus.exit_signal,
                ReplicaStatus.fail_reason,
                ReplicaStatus.last_seen,
            )
            .filter(ReplicaStatus.swarm_id == swarm_name)
            .order_by(ReplicaStatus.replica_id)
            .all()
        )

    return [
        {
            "replica_id": r.replica_id,
            "node": r.node,
            "port": r.port,
            "state": r.state,
            "exit_code": r.exit_code,
            "exit_signal": r.exit_signal,
            "fail_reason": r.fail_reason,
            "last_seen": r.last_seen.isoformat() if r.last_seen else None,
        }
        for r in rows
        if r.state in (ReplicaState.FAILED, ReplicaState.EXITED) or r.fail_reason
    ]
