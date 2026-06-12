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

from importlib.resources import files
import os
import re

# NOTE: alembic and sqlalchemy are imported lazily inside the functions below.
# Importing alembic costs ~2s, and the common case (DB already at head) can be
# decided with the cheap `*_fast` helpers without importing it at all.


def alembic_config_for(db_path: str):
    """
    Create an Alembic Config object for the given database path.
    """
    from alembic.config import Config

    cfg = Config()  # no .ini file required
    cfg.set_main_option(
        "script_location", str(files("domyn_swarm.core.state").joinpath("migrations"))
    )
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    return cfg


def upgrade_head(db_path: str):
    """
    Upgrade the database at the given path to the latest revision.
    """
    from alembic import command

    cfg = alembic_config_for(db_path)
    command.upgrade(cfg, "head")


def stamp_head(db_path: str):
    """
    Stamp the database at the given path with the latest revision without
    performing any migrations.
    """
    from alembic import command

    cfg = alembic_config_for(db_path)
    command.stamp(cfg, "head")


def get_current_rev(db_path: str) -> str | None:
    """
    Get the current revision of the database at the given path.
    Returns None if the database is unversioned.
    """
    from alembic.runtime.migration import MigrationContext
    from sqlalchemy import create_engine

    url = f"sqlite:///{os.path.abspath(db_path)}"
    engine = create_engine(url)

    with engine.connect() as conn:
        context = MigrationContext.configure(conn)
        return context.get_current_revision()


def get_head_rev(db_path: str) -> str | None:
    """
    Get the head revision of the database at the given path.
    """
    from alembic.script import ScriptDirectory

    cfg = alembic_config_for(db_path)
    script = ScriptDirectory.from_config(cfg)
    return script.get_current_head()


# --- Cheap, alembic-free fast paths (used to skip the ~2s alembic import when
# the DB is already at head, which is the common case). Both return None when
# the answer can't be determined unambiguously, so callers fall back to alembic.

_REVISION_RE = re.compile(r'^revision\s*(?::[^=]*)?=\s*["\']([^"\']+)["\']', re.M)
_DOWN_REVISION_RE = re.compile(r'^down_revision\s*(?::[^=]*)?=\s*["\']([^"\']+)["\']', re.M)


def current_rev_fast(db_path: str) -> str | None:
    """Read the current revision straight from SQLite, without importing alembic.

    Returns None if the DB is missing, unversioned, or has multiple heads (the
    caller should then defer to alembic).
    """
    import sqlite3

    if not os.path.exists(db_path):
        return None
    try:
        con = sqlite3.connect(db_path)
        try:
            rows = con.execute("SELECT version_num FROM alembic_version").fetchall()
        finally:
            con.close()
    except sqlite3.OperationalError:
        return None  # no alembic_version table → unversioned
    return rows[0][0] if len(rows) == 1 else None


def head_rev_fast() -> str | None:
    """Compute the head revision by scanning migration scripts, without alembic.

    Head is the revision that no other migration declares as its down_revision.
    Returns None if it can't be determined unambiguously.
    """
    versions = files("domyn_swarm.core.state").joinpath("migrations").joinpath("versions")
    revisions: set[str] = set()
    downs: set[str] = set()
    try:
        entries = list(versions.iterdir())
    except (FileNotFoundError, NotADirectoryError):
        return None
    for entry in entries:
        if not entry.name.endswith(".py") or entry.name == "__init__.py":
            continue
        text = entry.read_text(encoding="utf-8")
        m = _REVISION_RE.search(text)
        if m:
            revisions.add(m.group(1))
        downs.update(_DOWN_REVISION_RE.findall(text))
    heads = revisions - downs
    return next(iter(heads)) if len(heads) == 1 else None
