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

import importlib

import pytest


@pytest.fixture
def migrate_mod():
    """
    Import the migrations module once for reuse.
    Adjust the import path if your module lives elsewhere.
    """
    return importlib.import_module("domyn_swarm.core.state.migrate")


def test_upgrade_head_calls_alembic_upgrade(mocker, tmp_path, migrate_mod):
    db_path = tmp_path / "swarm.db"

    # alembic is imported lazily inside the function; patch it at the source.
    upgrade_mock = mocker.patch("alembic.command.upgrade")

    migrate_mod.upgrade_head(str(db_path))

    upgrade_mock.assert_called_once()
    cfg_arg, rev_arg = upgrade_mock.call_args[0]
    # Should call alembic.command.upgrade(cfg, "head")
    assert rev_arg == "head"
    # Config should have the right sqlalchemy.url
    assert cfg_arg.get_main_option("sqlalchemy.url") == f"sqlite:///{db_path}"


def test_stamp_head_calls_alembic_stamp(mocker, tmp_path, migrate_mod):
    db_path = tmp_path / "swarm.db"

    stamp_mock = mocker.patch("alembic.command.stamp")

    migrate_mod.stamp_head(str(db_path))

    stamp_mock.assert_called_once()
    cfg_arg, rev_arg = stamp_mock.call_args[0]
    # Should call alembic.command.stamp(cfg, "head")
    assert rev_arg == "head"
    assert cfg_arg.get_main_option("sqlalchemy.url") == f"sqlite:///{db_path}"


def test_get_current_rev_uses_migration_context(mocker, tmp_path, migrate_mod):
    db_path = tmp_path / "swarm.db"

    # Patch create_engine so we don't touch a real DB (imported lazily → patch source)
    engine_mock = mocker.patch("sqlalchemy.create_engine")
    engine = mocker.MagicMock()
    engine_mock.return_value = engine

    # Mock the connection returned by the context manager
    conn = mocker.MagicMock()
    # engine.connect() returns a context manager whose __enter__ returns conn
    engine.connect.return_value.__enter__.return_value = conn

    # Patch MigrationContext.configure(...) and its get_current_revision()
    configure_mock = mocker.patch("alembic.runtime.migration.MigrationContext.configure")
    ctx = mocker.MagicMock()
    ctx.get_current_revision.return_value = "abc123"
    configure_mock.return_value = ctx

    rev = migrate_mod.get_current_rev(str(db_path))

    engine_mock.assert_called_once()
    (url_arg,) = engine_mock.call_args[0]
    assert url_arg.startswith("sqlite:///")
    assert url_arg.endswith("/swarm.db")

    configure_mock.assert_called_once_with(conn)
    ctx.get_current_revision.assert_called_once_with()
    assert rev == "abc123"


def test_get_head_rev_uses_scriptdirectory(mocker, tmp_path, migrate_mod):
    db_path = tmp_path / "swarm.db"

    from alembic.config import Config

    fake_script = mocker.Mock()
    fake_script.get_current_head.return_value = "head_rev_456"

    from_config_mock = mocker.patch(
        "alembic.script.ScriptDirectory.from_config",
        return_value=fake_script,
    )

    rev = migrate_mod.get_head_rev(str(db_path))

    from_config_mock.assert_called_once()
    (cfg_arg,) = from_config_mock.call_args[0]
    assert isinstance(cfg_arg, Config)
    assert cfg_arg.get_main_option("sqlalchemy.url") == f"sqlite:///{db_path}"
    assert rev == "head_rev_456"


def test_head_rev_fast_matches_alembic(migrate_mod, tmp_path):
    """The alembic-free head scan agrees with alembic's ScriptDirectory."""
    db_path = tmp_path / "swarm.db"
    assert migrate_mod.head_rev_fast() == migrate_mod.get_head_rev(str(db_path))


def test_head_rev_fast_does_not_import_alembic(migrate_mod):
    """head_rev_fast must not pull in alembic (it's the whole point)."""
    import sys

    sys.modules.pop("alembic", None)
    migrate_mod.head_rev_fast()
    assert "alembic" not in sys.modules


def test_current_rev_fast_reads_sqlite(migrate_mod, tmp_path):
    """current_rev_fast reads alembic_version directly; None when missing/unversioned."""
    import sqlite3

    db_path = tmp_path / "swarm.db"
    assert migrate_mod.current_rev_fast(str(db_path)) is None  # missing file

    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE other (x)")  # exists but unversioned
    con.commit()
    assert migrate_mod.current_rev_fast(str(db_path)) is None

    con.execute("CREATE TABLE alembic_version (version_num VARCHAR(32))")
    con.execute("INSERT INTO alembic_version VALUES ('deadbeef')")
    con.commit()
    con.close()
    assert migrate_mod.current_rev_fast(str(db_path)) == "deadbeef"
