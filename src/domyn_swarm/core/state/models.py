from datetime import datetime

from sqlalchemy import DateTime, String, func
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
