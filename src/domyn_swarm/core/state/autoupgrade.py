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

from pathlib import Path

from rich.console import Console

from domyn_swarm.config.settings import get_settings
from domyn_swarm.core.state.state_manager import SwarmStateManager
from domyn_swarm.helpers.logger import setup_logger

from .migrate import get_current_rev, get_head_rev, upgrade_head

logger = setup_logger(__name__)

_DB_UPGRADED = False  # process-local guard

console = Console()


def ensure_db_up_to_date(*, noisy: bool = False) -> None:
    """
    Ensure the local swarm.db is at the latest Alembic revision.

    - If DOMYN_SWARM_SKIP_DB_UPGRADE=1 → do nothing.
    - If the DB file does not exist → create it and apply all migrations.
    - If it exists but is unversioned → treat as "from scratch" and upgrade.
    - If it's behind head → upgrade to head.
    - If it's already at head → no-op.

    This is intended to be called once per CLI invocation (guarded by _DB_UPGRADED).
    """
    global _DB_UPGRADED
    if _DB_UPGRADED:
        return

    settings = get_settings()

    if settings.skip_db_upgrade:
        logger.debug("Skipping DB auto-upgrade due to DOMYN_SWARM_SKIP_DB_UPGRADE=1")
        _DB_UPGRADED = True
        return

    db_path: Path = SwarmStateManager._get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_str = db_path.as_posix()

    # First-run: DB file does not exist → create + migrate
    if not db_path.exists():
        if noisy:
            console.print("[cyan]No swarm state DB found; creating and applying migrations…[/]")
        upgrade_head(db_str)
        if noisy:
            console.print("[green]DB schema initialized.[/]")
        _DB_UPGRADED = True
        return

    # Existing DB: figure out current vs head
    try:
        current = get_current_rev(db_str)  # may return None or raise if unversioned
    except Exception as e:  # pragma: no cover (defensive)
        logger.debug(f"Error reading current DB revision: {e!r}")
        current = None

    head = get_head_rev(db_str)

    # Already up to date
    if current == head:
        _DB_UPGRADED = True
        return

    # Needs upgrade
    if noisy:
        if current is None:
            console.print(f"[cyan]Unversioned DB found; upgrading to head ({head})…[/]")
        else:
            console.print(f"[cyan]Upgrading DB from {current} to {head}…[/]")

    upgrade_head(db_str)

    if noisy:
        console.print("[green]DB schema is up to date.[/]")

    _DB_UPGRADED = True
