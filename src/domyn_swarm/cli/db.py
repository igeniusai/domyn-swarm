import typer

db_app = typer.Typer(help="DB maintenance")


class _LazyLogger:
    """Logger proxy that avoids importing Rich logging during CLI discovery."""

    def __init__(self) -> None:
        self._logger = None

    def _get(self):
        from domyn_swarm.helpers.logger import setup_logger

        if self._logger is None:
            self._logger = setup_logger(__name__)
        return self._logger

    def __getattr__(self, name: str):
        return getattr(self._get(), name)


logger = _LazyLogger()


class _LazySwarmStateManager:
    """Proxy for state-manager class methods used by DB commands."""

    def list_all(self, *args, **kwargs):
        """List all saved swarm records."""
        from domyn_swarm.core.state.state_manager import SwarmStateManager

        return SwarmStateManager.list_all(*args, **kwargs)

    def delete_records(self, *args, **kwargs):
        """Delete saved swarm records by deployment name."""
        from domyn_swarm.core.state.state_manager import SwarmStateManager

        return SwarmStateManager.delete_records(*args, **kwargs)


SwarmStateManager = _LazySwarmStateManager()


def get_settings(*args, **kwargs):
    """Load environment settings lazily."""
    from domyn_swarm.config.settings import get_settings as load_settings

    return load_settings(*args, **kwargs)


def upgrade_head(*args, **kwargs):
    """Run Alembic upgrade lazily."""
    from domyn_swarm.core.state.migrate import upgrade_head as run_upgrade

    return run_upgrade(*args, **kwargs)


def stamp_head(*args, **kwargs):
    """Run Alembic stamp lazily."""
    from domyn_swarm.core.state.migrate import stamp_head as run_stamp

    return run_stamp(*args, **kwargs)


@db_app.command("upgrade")
def db_upgrade():
    db_path = (get_settings().home / "swarm.db").as_posix()
    upgrade_head(db_path)


@db_app.command("stamp")
def db_stamp():
    db_path = (get_settings().home / "swarm.db").as_posix()
    stamp_head(db_path)


@db_app.command("prune")
def db_prune(
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt before deleting records.",
    ),
) -> None:
    """Delete swarm records with FAILED, STOPPED, or UNKNOWN live status.

    Args:
        yes (bool): Skip confirmation prompt when True.
    """
    from domyn_swarm.core.swarm import DomynLLMSwarm
    from domyn_swarm.platform.protocols import ServingPhase

    to_be_deleted_phases = {ServingPhase.FAILED, ServingPhase.STOPPED, ServingPhase.UNKNOWN}
    records = SwarmStateManager.list_all()
    to_be_deleted: list[str] = []

    for rec in records:
        deployment_name = rec.get("deployment_name") or rec.get("name")
        if not deployment_name:
            continue
        try:
            swarm = DomynLLMSwarm.from_state(deployment_name=deployment_name)
            st = swarm._deployment.serving.status(swarm.serving_handle)  # type: ignore[attr-defined]
        except Exception as e:
            logger.debug(f"Status probe failed for {deployment_name}: {e}")
            to_be_deleted.append(deployment_name)
            continue
        if st.phase in to_be_deleted_phases:
            to_be_deleted.append(deployment_name)

    unique_to_be_deleted = sorted(set(to_be_deleted))
    if not unique_to_be_deleted:
        typer.echo("No dirty swarm records found.")
        return
    if not yes and not typer.confirm(f"Delete {len(unique_to_be_deleted)} swarm record(s)?"):
        raise typer.Exit(0)

    deleted = SwarmStateManager.delete_records(unique_to_be_deleted)
    typer.echo(f"Deleted {deleted} swarm record(s).")
