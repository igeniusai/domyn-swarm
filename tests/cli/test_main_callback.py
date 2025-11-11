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
def cli_main_mod():
    """
    Import the CLI main module where `app` and `main_callback` live.
    Adjust the import path if needed.
    """
    return importlib.import_module("domyn_swarm.cli.main")


class DummyCtx:
    def __init__(self, invoked_subcommand=None):
        self.invoked_subcommand = invoked_subcommand


def test_main_callback_skips_db_subcommand(mocker, cli_main_mod):
    """
    When the user invokes `domyn-swarm db ...`, we should NOT run
    the auto-upgrade helper (let the db subcommands manage migrations).
    """
    ensure_mock = mocker.patch.object(
        cli_main_mod, "ensure_db_up_to_date", autospec=True
    )

    ctx = DummyCtx(invoked_subcommand="db")

    cli_main_mod.main_callback(ctx)

    ensure_mock.assert_not_called()


def test_main_callback_runs_autoupgrade_for_other_subcommands(mocker, cli_main_mod):
    """
    For non-`db` subcommands, `main_callback` should call ensure_db_up_to_date(noisy=True).
    """
    ensure_mock = mocker.patch.object(
        cli_main_mod, "ensure_db_up_to_date", autospec=True
    )

    ctx = DummyCtx(invoked_subcommand="swarm")  # e.g. `domyn-swarm swarm list`

    cli_main_mod.main_callback(ctx)

    ensure_mock.assert_called_once_with(noisy=True)


def test_main_callback_runs_autoupgrade_when_subcommand_unknown(mocker, cli_main_mod):
    """
    If invoked_subcommand is None (e.g. bare `domyn-swarm`), we still want auto-upgrade.
    """
    ensure_mock = mocker.patch.object(
        cli_main_mod, "ensure_db_up_to_date", autospec=True
    )

    ctx = DummyCtx(invoked_subcommand=None)

    cli_main_mod.main_callback(ctx)

    ensure_mock.assert_called_once_with(noisy=True)
