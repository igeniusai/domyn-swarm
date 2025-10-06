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

"""State manager.

State is managed with a SQLite DB, located in the
domyn-swarm home directory.
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from domyn_swarm.config.slurm import SlurmConfig

if TYPE_CHECKING:
    from ..core.swarm import DomynLLMSwarm

import dataclasses

from domyn_swarm.config.backend import BackendConfig
from domyn_swarm.config.settings import get_settings
from domyn_swarm.platform.protocols import ServingHandle

from ..backends.compute.lepton import LeptonComputeBackend
from ..backends.compute.slurm import SlurmComputeBackend
from ..exceptions import JobNotFoundError
from ..helpers.logger import setup_logger

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
    """

    DB_NAME = "swarm.db"

    def __init__(self, swarm: "DomynLLMSwarm"):
        """Initialize the state manager.

        Args:
            swarm (DomynLLMSwarm): Swarm.
        """
        self.swarm: DomynLLMSwarm = swarm
        self._create_db_if_missing()

    def save(self, deployment_name: str) -> None:
        """Save a swarm state.

        Args:
            deployment_name (str): Deployment name.
        """
        record = self._get_record(deployment_name)
        db_path = self._get_db_path()
        query = _read_query("insert_record.sql")

        with sqlite3.connect(db_path) as conn:
            conn.execute(query, record)

        logger.debug(f"State saved for swarm {deployment_name}.")

    @classmethod
    def load(cls, deployment_name: str):
        """Load a swarm from the DB.

        Args:
            deployment_name (str): Deployment name.

        Returns:
            DomynLLMSwarm: Swarm.
        """
        from .. import DomynLLMSwarm

        db_path = cls._get_db_path()
        query = _read_query("select_swarm.sql")

        with sqlite3.connect(db_path) as cnx:
            cnx.row_factory = sqlite3.Row
            cursor = cnx.cursor()
            cursor.execute(query, {"deployment_name": deployment_name})
            record = cursor.fetchone()

        if not record:
            raise JobNotFoundError(deployment_name)

        swarm_dict = json.loads(record["swarm"])
        swarm_dict["cfg"] = json.loads(record["cfg"])
        handle_dict = json.loads(record["serving_handle"])

        swarm = DomynLLMSwarm.model_validate(swarm_dict)
        # TODO: fixme: should not be necessary
        # swarm.name = deployment_name
        serving_handle = ServingHandle(**handle_dict)
        swarm.serving_handle = serving_handle
        swarm._deployment._handle = serving_handle
        platform = swarm._platform
        if platform == "slurm":
            backend = cls._get_slurm_backend(
                handle=swarm.serving_handle, slurm_cfg=swarm.cfg.backend
            )
        elif platform == "lepton":
            backend = LeptonComputeBackend()
        else:
            raise ValueError(f"Unsupported platform: {platform}")

        swarm._deployment.compute = backend
        return swarm

    def delete_record(self, deployment_name: str) -> None:
        """Delete a record from the DB.

        Args:
            deployment_name (str): Deployment name.
        """
        db_path = self._get_db_path()
        query = _read_query("delete_record.sql")
        with sqlite3.connect(db_path) as cnx:
            cnx.execute(query, {"deployment_name": deployment_name})

    def _create_db_if_missing(self) -> None:
        """Create a new SQLite DB if it doesn't exist."""
        db_path = self._get_db_path()
        query = _read_query(fname="create_table.sql")
        with sqlite3.connect(db_path) as cnx:
            cnx.execute(query)

    @classmethod
    def _get_db_path(cls) -> Path:
        settings = get_settings()
        return settings.home / cls.DB_NAME

    def _get_record(self, deployment_name: str) -> dict[str, Any]:
        """Get a table record.

        Args:
            deployment_name (str): Deployment name.

        Returns:
            dict[str, Any]: Flattened record.
        """
        cfg = self.swarm.cfg.model_dump_json()
        swarm = self.swarm.model_dump_json(exclude={"cfg": True})
        handle = self.swarm.serving_handle
        if handle is None:
            raise ValueError("Null Serving handle")

        handle_dict = dataclasses.asdict(handle)
        serving_handle = json.dumps(handle_dict)

        return {
            "deployment_name": deployment_name,
            "swarm": swarm,
            "cfg": cfg,
            "serving_handle": serving_handle,
            "creation_dt": datetime.now(),
        }

    @classmethod
    def _get_slurm_backend(
        cls, handle: ServingHandle, slurm_cfg: BackendConfig | None
    ) -> SlurmComputeBackend:
        """Mostly done for the type checker.

        Args:
            handle (ServingHandle): Serving handle.
            slurm_cfg (BackendConfig | None): Slurm configs.

        Raises:
            ValueError: Null JobID.
            ValueError: Null LB info.

        Returns:
            SlurmComputeBackend: Slurm backend.
        """
        jobid = handle.meta.get("jobid")
        lb_jobid = handle.meta.get("lb_jobid")
        lb_node = handle.meta.get("lb_node")
        if jobid is None:
            raise ValueError("State file does not contain valid job IDs")
        if lb_jobid is None or lb_node is None:
            raise ValueError("State file does not contain valid LB job info")
        assert isinstance(slurm_cfg, SlurmConfig)

        return SlurmComputeBackend(cfg=slurm_cfg, lb_jobid=lb_jobid, lb_node=lb_node)
