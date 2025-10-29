from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def make_db_url(db_path: Path) -> str:
    return f"sqlite:///{db_path}"


def create_engine_for(db_path: Path):
    # check_same_thread=False lets us reuse connections in a CLI that may hand
    # off work across threads (Rich, Typer callbacks, etc.).
    return create_engine(
        make_db_url(db_path),
        future=True,
        echo=False,
        connect_args={"check_same_thread": False},
    )


def make_session_factory(db_path: Path):
    engine = create_engine_for(db_path)
    # DO NOT call Base.metadata.create_all() here; Alembic will manage schema.
    return sessionmaker(
        bind=engine, autoflush=False, autocommit=False, expire_on_commit=False
    )
