import typer

from domyn_swarm.config.settings import get_settings
from domyn_swarm.core.state.migrate import stamp_head, upgrade_head

db_app = typer.Typer(help="DB maintenance")


@db_app.command("upgrade")
def db_upgrade():
    db_path = (get_settings().home / "swarm.db").as_posix()
    upgrade_head(db_path)


@db_app.command("stamp")
def db_stamp():
    db_path = (get_settings().home / "swarm.db").as_posix()
    stamp_head(db_path)
