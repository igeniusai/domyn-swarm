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

    upgrade_mock = mocker.patch.object(migrate_mod.command, "upgrade")

    migrate_mod.upgrade_head(str(db_path))

    upgrade_mock.assert_called_once()
    cfg_arg, rev_arg = upgrade_mock.call_args[0]
    # Should call alembic.command.upgrade(cfg, "head")
    assert rev_arg == "head"
    # Config should have the right sqlalchemy.url
    assert cfg_arg.get_main_option("sqlalchemy.url") == f"sqlite:///{db_path}"


def test_stamp_head_calls_alembic_stamp(mocker, tmp_path, migrate_mod):
    db_path = tmp_path / "swarm.db"

    stamp_mock = mocker.patch.object(migrate_mod.command, "stamp")

    migrate_mod.stamp_head(str(db_path))

    stamp_mock.assert_called_once()
    cfg_arg, rev_arg = stamp_mock.call_args[0]
    # Should call alembic.command.stamp(cfg, "head")
    assert rev_arg == "head"
    assert cfg_arg.get_main_option("sqlalchemy.url") == f"sqlite:///{db_path}"


def test_get_current_rev_uses_migration_context(mocker, tmp_path, migrate_mod):
    db_path = tmp_path / "swarm.db"

    # Patch create_engine so we don't touch a real DB
    engine_mock = mocker.patch.object(migrate_mod, "create_engine")
    engine = mocker.MagicMock()
    engine_mock.return_value = engine

    # Mock the connection returned by the context manager
    conn = mocker.MagicMock()
    # engine.connect() returns a context manager whose __enter__ returns conn
    engine.connect.return_value.__enter__.return_value = conn

    # Patch MigrationContext.configure(...) and its get_current_revision()
    configure_mock = mocker.patch.object(migrate_mod.MigrationContext, "configure")
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

    fake_script = mocker.Mock()
    fake_script.get_current_head.return_value = "head_rev_456"

    from_config_mock = mocker.patch.object(
        migrate_mod.ScriptDirectory,
        "from_config",
        return_value=fake_script,
    )

    rev = migrate_mod.get_head_rev(str(db_path))

    from_config_mock.assert_called_once()
    (cfg_arg,) = from_config_mock.call_args[0]
    assert isinstance(cfg_arg, migrate_mod.Config)
    assert cfg_arg.get_main_option("sqlalchemy.url") == f"sqlite:///{db_path}"
    assert rev == "head_rev_456"
