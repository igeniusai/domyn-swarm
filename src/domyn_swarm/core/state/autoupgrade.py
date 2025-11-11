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

# domyn_swarm/db/autoupgrade.py (or in migrations.py if you prefer)

from __future__ import annotations

import typer
from rich.console import Console

from domyn_swarm.config.settings import get_settings
from domyn_swarm.helpers.logger import setup_logger

from .migrate import get_current_rev, get_head_rev, upgrade_head

logger = setup_logger(__name__)

_DB_UPGRADED = False  # process-local guard


def ensure_db_up_to_date(*, noisy: bool = True) -> None:
    """
    Idempotently ensure the swarm.db schema is at Alembic HEAD.

    - Runs at most once per process.
    - If current != head, runs `upgrade_head`.
    - If anything fails, logs and aborts the CLI.
    """
    global _DB_UPGRADED
    if _DB_UPGRADED:
        return

    settings = get_settings()

    # Allow tests or power users to bypass auto-upgrade if really needed.
    if settings.skip_db_upgrade:
        logger.debug("Skipping DB auto-upgrade due to DOMYN_SWARM_SKIP_DB_UPGRADE=1")
        _DB_UPGRADED = True
        return

    db_path = (settings.home / "swarm.db").as_posix()

    try:
        current = get_current_rev(
            db_path
        )  # your helper: returns current revision or None
        head = get_head_rev(db_path)  # your helper: returns Alembic head revision

        if current == head:
            logger.debug("State DB already at latest migration (rev=%s)", head)
            _DB_UPGRADED = True
            return

        console = Console(stderr=True)

        if noisy:
            console.print("[yellow]New database version detected, upgrading…[/]")

        upgrade_head(db_path)

        if noisy:
            console.print("[green]Database schema upgraded.[/]")

        _DB_UPGRADED = True

    except Exception as e:  # noqa: BLE001
        logger.error("Failed to upgrade state DB: %s", e)
        # For CLI usage, failing hard is usually better than silently corrupting state.
        raise typer.Exit(code=1) from e
