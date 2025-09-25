"""Test the state persistence."""

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from domyn_swarm import DomynLLMSwarm, DomynLLMSwarmConfig
from domyn_swarm.config.slurm import SlurmConfig, SlurmEndpointConfig
from domyn_swarm.core.state import SwarmStateManager, _read_query
from domyn_swarm.exceptions import JobNotFoundError
from domyn_swarm.platform.protocols import ServingHandle


def test_read_query() -> None:
    """Check queries are properly read."""
    query = _read_query("create_table.sql")
    assert isinstance(query, str)
    assert "CREATE TABLE" in query


class TestSwarmStateManager:
    """Test the tate persistence."""

    @pytest.fixture
    def swarm(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DomynLLMSwarm:
        """Swarm.

        Returns:
            DomynLLMSwarm: Swarm.
        """

        def mock_db_path(_x=None) -> Path:
            """Mock DB path.

            Args:
                _x (Any, optional): mocked argument. Defaults to None.

            Returns:
                Path: Temporary DB path.
            """
            return tmp_path / SwarmStateManager.DB_NAME

        monkeypatch.setattr(SwarmStateManager, "_get_db_path", mock_db_path)
        return DomynLLMSwarm(
            name="swarm",
            cfg=DomynLLMSwarmConfig(
                name="fake",
                hf_home=".",
                model="Qwen/Qwen3-32B",
                revision=None,
                replicas=1,
                gpus_per_node=1,
                gpus_per_replica=1,
                replicas_per_node=1,
                nodes=1,
                port=1000,
                cpus_per_task=2,
                mem_per_cpu="1GB",
                wait_endpoint_s=1200,
                backend=SlurmConfig(
                    type="slurm",
                    requires_ray=False,
                    endpoint=SlurmEndpointConfig(
                        cpus_per_task=1,
                        mem="1GB",
                        threads_per_core=1,
                        wall_time="24:00:00",
                        enable_proxy_buffering=True,
                        nginx_timeout="60s",
                    ),
                ),
            ),
            serving_handle=ServingHandle(
                id="1234",
                url="http://lb:9003",
                meta=dict(
                    lb_jobid=1234,
                    jobid=5678,
                    lb_node="lrdn4759",
                    port=9003,
                    name="swarm",
                ),
            ),
            endpoint="http://lb:9003",
            delete_on_exit=True,
        )

    @pytest.fixture
    def state_manager(self, swarm: DomynLLMSwarm) -> SwarmStateManager:
        """Swarm state manager.

        Args:
            swarm (DomynLLMSwarm): Swarm.

        Returns:
            SwarmStateManager: State manager.
        """
        return swarm._state_mgr

    def test_get_db_path(self) -> None:
        """The DB name must be correct."""
        db_path = SwarmStateManager._get_db_path()
        assert db_path.name == SwarmStateManager.DB_NAME

    def test_create_missing_table(self, state_manager: SwarmStateManager) -> None:
        """Test the DB creation.

        A DB with the swarm table must be created when missing.

        Args:
            state_manager (SwarmStateManager): State manager.
        """
        db_path = state_manager._get_db_path()
        db_path.unlink(missing_ok=True)
        state_manager._create_db_if_missing()
        with sqlite3.connect(db_path) as cnx:
            cursor = cnx.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='swarm';"
            )
            # one-element tuple
            name = cursor.fetchone()

        assert name[0] == "swarm"

    def test_create_already_present_table(
        self, state_manager: SwarmStateManager
    ) -> None:
        """The table must be left untouched if it exists.

        Args:
            state_manager (SwarmStateManager): State manager.
        """
        state_manager._create_db_if_missing()
        state_manager._create_db_if_missing()
        db_path = state_manager._get_db_path()
        with sqlite3.connect(db_path) as cnx:
            cursor = cnx.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='swarm';"
            )
            # one-element tuple
            name = cursor.fetchone()

        assert name[0] == "swarm"

    def test_get_record(self, state_manager: SwarmStateManager) -> None:
        """Check the table record.

        Args:
            state_manager (SwarmStateManager): State manager.
        """
        data = state_manager._get_record("fake")
        assert data["deployment_name"] == "fake"
        assert isinstance(data["swarm"], str)
        assert isinstance(data["cfg"], str)
        assert isinstance(data["serving_handle"], str)
        assert isinstance(data["creation_dt"], datetime)

    def test_get_record_null_handle(self, state_manager: SwarmStateManager) -> None:
        """The handle must be valid.

        Args:
            state_manager (SwarmStateManager): State manager.
        """
        state_manager.swarm.serving_handle = None
        with pytest.raises(ValueError):
            state_manager._get_record("fake")

    def test_save(self, state_manager: SwarmStateManager) -> None:
        """Check the record is saved in the DB.

        Args:
            state_manager (SwarmStateManager): State manager.
        """
        state_manager._create_db_if_missing()
        state_manager.save(deployment_name="name")
        db_path = state_manager._get_db_path()
        with sqlite3.connect(db_path) as cnx:
            cursor = cnx.cursor()
            cursor.execute("SELECT * FROM swarm WHERE deployment_name='name';")
            row = cursor.fetchall()

        assert len(row) == 1

    def test_load_slurm(self, state_manager: SwarmStateManager) -> None:
        """Test the swarm initialization from a record.

        Args:
            state_manager (SwarmStateManager): State manager.
        """
        state_manager._create_db_if_missing()
        state_manager.save(deployment_name="fake")

        swarm = SwarmStateManager.load(deployment_name="fake")
        assert isinstance(swarm, DomynLLMSwarm)

    def test_load_null_jobid(self, state_manager: SwarmStateManager) -> None:
        """Null Job ID.

        Args:
            state_manager (SwarmStateManager): State manager.
        """
        state_manager.swarm.serving_handle.meta["jobid"] = None
        state_manager._create_db_if_missing()
        state_manager.save(deployment_name="fake")

        with pytest.raises(ValueError, match="job IDs"):
            SwarmStateManager.load(deployment_name="fake")

    def test_load_null_lb_jobid(self, state_manager: SwarmStateManager) -> None:
        """Null LB Job ID.

        Args:
            state_manager (SwarmStateManager): State manager.
        """
        state_manager.swarm.serving_handle.meta["lb_jobid"] = None
        state_manager._create_db_if_missing()
        state_manager.save(deployment_name="fake")

        with pytest.raises(ValueError, match="LB job info"):
            SwarmStateManager.load(deployment_name="fake")

    def test_load_null_lb_node(self, state_manager: SwarmStateManager) -> None:
        """Null LB Node.

        Args:
            state_manager (SwarmStateManager): State manager.
        """
        state_manager.swarm.serving_handle.meta["lb_node"] = None
        state_manager._create_db_if_missing()
        state_manager.save(deployment_name="fake")

        with pytest.raises(ValueError, match="LB job info"):
            SwarmStateManager.load(deployment_name="fake")

    def test_load_unknown_job(self, state_manager: SwarmStateManager) -> None:
        """Exception is raised if the swarm is not found in the DB.

        Args:
            state_manager (SwarmStateManager): State manager.
        """
        with pytest.raises(JobNotFoundError, match="fake"):
            SwarmStateManager.load(deployment_name="fake")

    def test_delete_record(self, state_manager: SwarmStateManager) -> None:
        """Test the record deletion from the DB.

        Args:
            state_manager (SwarmStateManager): State manager.
        """
        state_manager._create_db_if_missing()
        state_manager.save(deployment_name="fake")
        state_manager.swarm.delete_record(deployment_name="fake")
        db_path = state_manager._get_db_path()

        with sqlite3.connect(db_path) as cnx:
            cursor = cnx.cursor()
            cursor.execute("SELECT * FROM swarm WHERE deployment_name='fake';")
            rows = cursor.fetchall()

        assert not rows
