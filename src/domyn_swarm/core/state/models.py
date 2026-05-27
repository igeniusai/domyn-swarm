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

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    pass


class SwarmRecord(Base):
    __tablename__ = "swarm"

    deployment_name: Mapped[str] = mapped_column(String, primary_key=True)
    swarm: Mapped[dict] = mapped_column(JSON, nullable=False)
    cfg: Mapped[dict] = mapped_column(JSON, nullable=False)
    serving_handle: Mapped[dict] = mapped_column(JSON, nullable=False)
    creation_dt: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        server_default=func.current_timestamp(),
        nullable=False,
    )


class ReplicaStatus(Base):
    __tablename__ = "replica_status"

    swarm_id: Mapped[str] = mapped_column(String, primary_key=True)
    replica_id: Mapped[int] = mapped_column(Integer, primary_key=True)

    node: Mapped[str | None] = mapped_column(String, nullable=True)
    port: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)

    state: Mapped[str | None] = mapped_column(
        String, nullable=True
    )  # e.g. "Starting", "Running", "Failed", "Exited"
    http_ready: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    exit_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    exit_signal: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fail_reason: Mapped[str | None] = mapped_column(String, nullable=True)

    agent_version: Mapped[str | None] = mapped_column(String, nullable=True)
    last_seen: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        server_default=func.current_timestamp(),
        nullable=False,
    )


class JobRecord(Base):
    """Database record for a compute job targeting a swarm.

    This table lives in the local `swarm.db` state database and is meant to provide
    reconnectable job semantics across process boundaries.
    """

    __tablename__ = "jobs"

    job_id: Mapped[str] = mapped_column(String, primary_key=True)
    deployment_name: Mapped[str] = mapped_column(
        String,
        ForeignKey("swarm.deployment_name"),
        nullable=False,
        index=True,
    )

    provider: Mapped[str] = mapped_column(String, nullable=False)  # e.g. "slurm"
    kind: Mapped[str] = mapped_column(String, nullable=False)  # e.g. "step" | "sbatch"

    status: Mapped[str] = mapped_column(String, nullable=False)
    raw_status: Mapped[str | None] = mapped_column(String, nullable=True)

    external_id: Mapped[str | None] = mapped_column(String, nullable=True)  # e.g. "12345.0"
    name: Mapped[str | None] = mapped_column(String, nullable=True)

    command: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    resources: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    log_paths: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    error: Mapped[str | None] = mapped_column(String, nullable=True)

    creation_dt: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        server_default=func.current_timestamp(),
        nullable=False,
    )
    update_dt: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        server_default=func.current_timestamp(),
        nullable=False,
    )
