# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

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


@pytest.mark.parametrize("subcommand", ["db", "init", "version"])
def test_main_callback_skips_subcommands_without_state_access(mocker, cli_main_mod, subcommand):
    """
    Commands without swarm state access should not run the auto-upgrade helper.
    """
    ensure_mock = mocker.patch.object(cli_main_mod, "ensure_db_up_to_date", autospec=True)

    ctx = DummyCtx(invoked_subcommand=subcommand)

    cli_main_mod.main_callback(ctx)

    ensure_mock.assert_not_called()


def test_main_callback_runs_autoupgrade_for_other_subcommands(mocker, cli_main_mod):
    """
    For non-`db` subcommands, `main_callback` should call ensure_db_up_to_date(noisy=True).
    """
    ensure_mock = mocker.patch.object(cli_main_mod, "ensure_db_up_to_date", autospec=True)

    ctx = DummyCtx(invoked_subcommand="swarm")  # e.g. `domyn-swarm swarm list`

    cli_main_mod.main_callback(ctx)

    ensure_mock.assert_called_once_with(noisy=True)


def test_main_callback_runs_autoupgrade_when_subcommand_unknown(mocker, cli_main_mod):
    """
    If invoked_subcommand is None (e.g. bare `domyn-swarm`), we still want auto-upgrade.
    """
    ensure_mock = mocker.patch.object(cli_main_mod, "ensure_db_up_to_date", autospec=True)

    ctx = DummyCtx(invoked_subcommand=None)

    cli_main_mod.main_callback(ctx)

    ensure_mock.assert_called_once_with(noisy=True)
