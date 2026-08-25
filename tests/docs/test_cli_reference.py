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

"""The generated CLI reference must cover every command, including lazy ones.

``LazyGroup`` in ``domyn_swarm.cli.main`` imports sub-apps on demand, so a
refactor there can silently empty the CLI reference. These tests fail loudly
instead.
"""

from __future__ import annotations

from pathlib import Path

from cli_app import cli
import click
import pytest

from domyn_swarm.cli.main import _LAZY_SUBAPPS

EXPECTED_SUBCOMMANDS: dict[str, set[str]] = {
    "job": {"submit", "submit-script", "list", "status", "wait", "cancel"},
    "swarm": {"list", "describe"},
    "db": {"upgrade", "stamp", "prune"},
    "init": {"defaults"},
    "pool": {"pool"},
}

EXPECTED_ROOT_COMMANDS = {"version", "up", "status", "down"}


def test_warmed_group_registers_every_lazy_subapp() -> None:
    missing = sorted(set(_LAZY_SUBAPPS) - set(cli.commands))
    assert not missing, (
        f"lazy sub-apps missing from the warmed group: {missing}. "
        "cli_app._warmed_group() must resolve every name from list_commands()."
    )


def test_warmed_group_keeps_the_eager_root_commands() -> None:
    assert set(cli.commands) >= EXPECTED_ROOT_COMMANDS


@pytest.mark.parametrize(
    ("parent", "children"),
    sorted(EXPECTED_SUBCOMMANDS.items()),
    ids=sorted(EXPECTED_SUBCOMMANDS),
)
def test_subcommands_are_reachable(parent: str, children: set[str]) -> None:
    group = cli.commands[parent]
    assert isinstance(group, click.Group), f"{parent} is not a group"
    listed = set(group.list_commands(click.Context(group)))
    missing = sorted(children - listed)
    assert not missing, f"{parent} is missing subcommands: {missing}"


def test_cli_page_wraps_the_directive_in_eval_rst() -> None:
    """sphinx-click emits reStructuredText, which MyST renders literally.

    Invoked as a bare MyST directive the whole options tree lands on the page as
    visible ``.. option::`` text and no flag is actually documented. Wrapping it
    in ``eval-rst`` hands the nested content to the RST parser.
    """
    page = Path(__file__).resolve().parents[2] / "docs" / "reference" / "cli.md"
    text = page.read_text(encoding="utf-8")
    assert "```{eval-rst}" in text, "the click directive must be inside an eval-rst block"
    assert ".. click:: cli_app:cli" in text


def test_root_carries_typers_completion_options() -> None:
    """The root must be built with ``get_command``, not ``get_group``.

    ``typer.main.get_group`` returns the command tree without Typer's own
    ``--install-completion`` / ``--show-completion`` options, so a reference built
    from it documents every command yet silently omits two global flags.
    """
    params = {p.name for p in cli.params}
    assert {"install_completion", "show_completion"} <= params
