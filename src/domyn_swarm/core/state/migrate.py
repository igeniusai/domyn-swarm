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

from importlib.resources import files
import os

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine


def alembic_config_for(db_path: str) -> Config:
    """
    Create an Alembic Config object for the given database path.
    """
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
    cfg = alembic_config_for(db_path)
    command.upgrade(cfg, "head")


def stamp_head(db_path: str):
    """
    Stamp the database at the given path with the latest revision without
    performing any migrations.
    """
    cfg = alembic_config_for(db_path)
    command.stamp(cfg, "head")


def get_current_rev(db_path: str) -> str | None:
    """
    Get the current revision of the database at the given path.
    Returns None if the database is unversioned.
    """
    url = f"sqlite:///{os.path.abspath(db_path)}"
    engine = create_engine(url)

    with engine.connect() as conn:
        context = MigrationContext.configure(conn)
        return context.get_current_revision()


def get_head_rev(db_path: str) -> str | None:
    """
    Get the head revision of the database at the given path.
    """
    cfg = alembic_config_for(db_path)
    script = ScriptDirectory.from_config(cfg)
    return script.get_current_head()
