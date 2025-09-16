"""State manager.

State is managed with a SQLite DB, located in the
domyn-swarm home directory.
"""

import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from domyn_swarm import DomynLLMSwarm

from domyn_swarm import utils
from domyn_swarm.exceptions import JobNotFoundError
from domyn_swarm.helpers.logger import setup_logger

logger = setup_logger(__name__, level=logging.INFO)


def _read_query(fname: str) -> str:
    """Read a query from a .sql file.

    Args:
        fname (str): Query file name.

    Returns:
        str: Query
    """
    path = Path("./queries") / fname
    with path.open("r") as fquery:
        return fquery.read()


class SwarmStateManager:
    """State manager

    Used to manage persistence through a SQLite DB. The DB
    contains one table named `swarm`.

    Attributes:
        DB_NAME (str): (classattribute) SQLite DB name.
        swarm (DomynLLMSwarm): Swarm.
        db_path (Path): Path to the DB.
    """

    DB_NAME = "swarm.db"

    def __init__(self, swarm: "DomynLLMSwarm"):
        """Initialize the state manager.

        Args:
            swarm (DomynLLMSwarm): Swarm.
        """
        self.swarm = swarm
        self.db_path = utils.EnvPath(self.swarm.cfg.home_directory) / self.DB_NAME
        self._create_db_if_missing()

    def save(self) -> None:
        """Save a swarm state."""
        data = self._get_flat_data_dict()
        query = _read_query("insert_record.sql")

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(query, data)

        logger.debug(f"State saved for job {self.swarm.jobid}.")

    @classmethod
    def load(cls, jobid: int, home_directory: Path) -> "DomynLLMSwarm":
        """Load a swarm from the DB.

        Args:
            jobid (int): Job ID (primary key).
            home_directory (Path): Domyn-swarm home directory.

        Raises:
            JobNotFoundError: A job with the same ID is not found
                in the DB.

        Returns:
            DomynLLMSwarm: Swarm.
        """
        from domyn_swarm import DomynLLMSwarm

        select_cfg = _read_query("select_cfg.sql")
        select_driver = _read_query("select_driver.sql")
        select_swarm = _read_query("select_swarm.sql")

        with sqlite3.connect(home_directory / cls.DB_NAME) as cnx:
            cnx.row_factory = sqlite3.Row
            cursor = cnx.cursor()
            cursor.execute(select_cfg, {"jobid": jobid})
            cfg_row = cursor.fetchone()
            cursor.execute(select_driver, {"jobid": jobid})
            driver_row = cursor.fetchone()
            cursor.execute(select_swarm, {"jobid": jobid})
            swarm_row = cursor.fetchone()

        if not swarm_row:
            raise JobNotFoundError(jobid)

        swarm_dict = dict(swarm_row)
        swarm_dict["cfg"] = dict(cfg_row)
        swarm_dict["cfg"]["driver"] = dict(driver_row)

        return DomynLLMSwarm.model_validate(swarm_dict)

    def update_lb_data(self) -> None:
        """Update the load balancer info."""
        query = _read_query(fname="update_lb.sql")
        data = {
            "jobid": self.swarm.jobid,
            "lb_port": self.swarm.cfg.lb_port,
            "lb_node": self.swarm.lb_node,
            "endpoint": self.swarm.endpoint,
        }
        with sqlite3.connect(self.db_path) as cnx:
            cnx.execute(query, data)

    def delete_record(self) -> None:
        """Delete a record from the DB."""
        query = _read_query("delete_record.sql")
        with sqlite3.connect(self.db_path) as cnx:
            cnx.execute(query, {"jobid": self.swarm.jobid})

    def _create_db_if_missing(self) -> None:
        """Create a new SQLite DB if it doesn't exist."""
        query = _read_query(fname="create_table.sql")
        with sqlite3.connect(self.db_path) as cnx:
            cnx.execute(query)

    def _get_flat_data_dict(self) -> dict[str, Any]:
        """Flatten configurations.

        JSON configuration must be flattened to be placed in
        a record of the DB. a 'driver_' prefix is added to
        driver data to avoid field duplication.

        Returns:
            dict[str, Any]: Flattened record.
        """
        cfg = self.swarm.cfg.model_dump(mode="json", exclude={"driver": True})
        model = self.swarm.model_dump(mode="json", exclude={"cfg": True})
        driver = self.swarm.cfg.driver.model_dump()
        driver = {f"driver_{k}": v for k, v in driver.items()}

        return model | driver | cfg
