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

from domyn_swarm.core.state.state_manager import SwarmStateManager


@pytest.fixture(autouse=True)
def reset_db_upgraded(monkeypatch):
    """
    Reset the module-level _DB_UPGRADED flag before each test so that
    ensure_db_up_to_date actually runs its logic.
    """
    mod = importlib.import_module("domyn_swarm.core.state.autoupgrade")
    monkeypatch.setattr(mod, "_DB_UPGRADED", False)
    yield


@pytest.fixture
def autoupgrade_mod():
    """Convenience fixture to import the autoupgrade module."""
    return importlib.import_module("domyn_swarm.core.state.autoupgrade")


@pytest.fixture
def tmp_db_path(tmp_path, monkeypatch):
    """
    Patch SwarmStateManager._get_db_path to point to a temp file under tmp_path.
    This keeps the DB location consistent between the code under test and
    the assertions in the tests.
    """
    db_path = tmp_path / SwarmStateManager.DB_NAME

    def mock_get_db_path(cls):
        return db_path

    monkeypatch.setattr(SwarmStateManager, "_get_db_path", classmethod(mock_get_db_path))
    return db_path


def test_ensure_db_up_to_date_no_upgrade_when_at_head(
    autoupgrade_mod, tmp_db_path, monkeypatch, mocker
):
    """
    If the DB exists and current == head, no upgrade should be performed.
    """
    monkeypatch.delenv("DOMYN_SWARM_SKIP_DB_UPGRADE", raising=False)

    tmp_db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_db_path.touch()

    upgrade_mock = mocker.patch.object(autoupgrade_mod, "upgrade_head")
    mocker.patch.object(autoupgrade_mod, "get_current_rev", return_value="rev1")
    mocker.patch.object(autoupgrade_mod, "get_head_rev", return_value="rev1")

    autoupgrade_mod.ensure_db_up_to_date(noisy=False)

    upgrade_mock.assert_not_called()


def test_ensure_db_up_to_date_upgrades_when_out_of_date(
    autoupgrade_mod, tmp_db_path, monkeypatch, mocker
):
    """
    If the DB exists and current != head, upgrade_head should be called.
    """
    monkeypatch.delenv("DOMYN_SWARM_SKIP_DB_UPGRADE", raising=False)

    tmp_db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_db_path.touch()

    upgrade_mock = mocker.patch.object(autoupgrade_mod, "upgrade_head")
    mocker.patch.object(autoupgrade_mod, "get_current_rev", return_value="old_rev")
    mocker.patch.object(autoupgrade_mod, "get_head_rev", return_value="new_rev")

    autoupgrade_mod.ensure_db_up_to_date(noisy=False)

    expected_db = tmp_db_path.as_posix()
    upgrade_mock.assert_called_once_with(expected_db)


def test_ensure_db_up_to_date_respects_skip_env(
    autoupgrade_mod, tmp_db_path, disable_autoupgrade, mocker
):
    """
    If DOMYN_SWARM_SKIP_DB_UPGRADE=1, ensure_db_up_to_date should be a no-op
    (other than setting _DB_UPGRADED).
    """
    # disable_autoupgrade fixture already sets DOMYN_SWARM_SKIP_DB_UPGRADE=1

    upgrade_mock = mocker.patch.object(autoupgrade_mod, "upgrade_head")
    current_mock = mocker.patch.object(autoupgrade_mod, "get_current_rev")
    head_mock = mocker.patch.object(autoupgrade_mod, "get_head_rev")

    autoupgrade_mod.ensure_db_up_to_date(noisy=True)

    upgrade_mock.assert_not_called()
    current_mock.assert_not_called()
    head_mock.assert_not_called()


def test_main_callback_triggers_autoupgrade_for_non_db_commands(monkeypatch, mocker, tmp_db_path):
    """
    main_callback should call ensure_db_up_to_date for normal commands,
    but not when the 'db' subcommand is invoked.
    """
    from domyn_swarm.cli import main as cli_main

    # Ensure env doesn't skip
    monkeypatch.delenv("DOMYN_SWARM_SKIP_DB_UPGRADE", raising=False)

    # Patch SwarmStateManager DB path to our tmp path (for consistency)
    def mock_get_db_path(cls):
        return tmp_db_path

    monkeypatch.setattr(SwarmStateManager, "_get_db_path", classmethod(mock_get_db_path))

    ensure_mock = mocker.patch.object(cli_main, "ensure_db_up_to_date")

    class Ctx:
        invoked_subcommand = "up"

    ctx = Ctx()
    cli_main.main_callback(ctx)  # type: ignore[arg-type]
    ensure_mock.assert_called_once_with(noisy=True)

    # Now simulate 'db' subcommand, which should skip autoupgrade
    ensure_mock.reset_mock()
    ctx.invoked_subcommand = "db"
    cli_main.main_callback(ctx)  # type: ignore[arg-type]
    ensure_mock.assert_not_called()
