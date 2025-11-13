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
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
