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

"""Expose a fully-resolved Click group for ``sphinx-click``.

``domyn_swarm.cli.main.LazyGroup`` imports its heavy sub-apps only when a
command is requested, so ``group.commands`` is empty until something asks. Any
introspection that reads ``commands`` directly would therefore miss ``job``,
``pool``, ``init``, ``swarm`` and ``db``. Resolving every name up front makes the
generated reference complete regardless of how the documenter traverses.
"""

from __future__ import annotations

import click
import typer.main

from domyn_swarm.cli.main import app


def _warmed_group() -> click.Group:
    """Return the root Click group with every lazy sub-app resolved."""
    # get_command, not get_group: only get_command attaches Typer's own
    # --install-completion / --show-completion options to the root, and a
    # reference that omits them would be missing two real global flags.
    group = typer.main.get_command(app)
    if not isinstance(group, click.Group):
        raise RuntimeError("the root CLI is expected to be a group of commands")
    ctx = click.Context(group)
    for name in group.list_commands(ctx):
        resolved = group.get_command(ctx, name)
        if resolved is None:  # pragma: no cover - defensive
            raise RuntimeError(
                f"command {name!r} was listed but could not be resolved; "
                "the CLI reference would be incomplete"
            )
    return group


cli = _warmed_group()
