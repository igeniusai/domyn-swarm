# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
import threading

from rich.console import Console

from domyn_swarm.config.settings import get_settings
from domyn_swarm.core.state.state_manager import SwarmStateManager
from domyn_swarm.helpers.logger import setup_logger

from .migrate import current_rev_fast, get_current_rev, get_head_rev, head_rev_fast, upgrade_head

logger = setup_logger(__name__)

_DB_UPGRADED = False  # process-local guard
_DB_UPGRADE_LOCK = threading.Lock()

console = Console()


def ensure_db_up_to_date(*, noisy: bool = False) -> None:
    """Upgrade the local state database to the latest Alembic revision.

    Args:
        noisy: Whether to print migration progress to the console.
    """
    global _DB_UPGRADED
    if _DB_UPGRADED:
        return

    with _DB_UPGRADE_LOCK:
        # Recheck after acquiring the lock because another thread may have upgraded the DB.
        if _DB_UPGRADED:
            return

        settings = get_settings()

        if settings.skip_db_upgrade:
            logger.debug("Skipping DB auto-upgrade due to DOMYN_SWARM_SKIP_DB_UPGRADE=1")
            _DB_UPGRADED = True
            return

        db_path: Path = SwarmStateManager._resolve_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_str = db_path.as_posix()

        if not db_path.exists():
            if noisy:
                console.print("[cyan]No swarm state DB found; creating and applying migrations…[/]")
            upgrade_head(db_str)
            if noisy:
                console.print("[green]DB schema initialized.[/]")
            _DB_UPGRADED = True
            return

        # Fast path: decide "already at head" with a cheap SQLite read + a script
        # scan, avoiding the ~2s alembic import in the common case.
        fast_current = current_rev_fast(db_str)
        fast_head = head_rev_fast()
        if fast_head is not None and fast_current == fast_head:
            _DB_UPGRADED = True
            return

        try:
            current = get_current_rev(db_str)  # may return None or raise if unversioned
        except Exception as e:  # pragma: no cover (defensive)
            logger.debug(f"Error reading current DB revision: {e!r}")
            current = None

        head = get_head_rev(db_str)

        if current == head:
            _DB_UPGRADED = True
            return

        if noisy:
            if current is None:
                console.print(f"[cyan]Unversioned DB found; upgrading to head ({head})…[/]")
            else:
                console.print(f"[cyan]Upgrading DB from {current} to {head}…[/]")

        upgrade_head(db_str)

        if noisy:
            console.print("[green]DB schema is up to date.[/]")

        _DB_UPGRADED = True
