from importlib.resources import files

from alembic import command
from alembic.config import Config


def alembic_config_for(db_path: str) -> Config:
    cfg = Config()  # no .ini file required
    cfg.set_main_option(
        "script_location", str(files("domyn_swarm.core.state").joinpath("migrations"))
    )
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    return cfg


def upgrade_head(db_path: str):
    cfg = alembic_config_for(db_path)
    command.upgrade(cfg, "head")


def stamp_head(db_path: str):
    cfg = alembic_config_for(db_path)
    command.stamp(cfg, "head")
