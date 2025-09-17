"""Test the state persistence."""

import sqlite3
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
    def swarm(self, tmp_path: Path) -> DomynLLMSwarm:
        """Swarm.

        Args:
            tmp_path (Path): Temporary path to store the DB.

        Returns:
            DomynLLMSwarm: Swarm.
        """
        return DomynLLMSwarm(
            name="swarm",
            cfg=DomynLLMSwarmConfig(
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
        return SwarmStateManager(swarm=swarm)

    def test_create_missing_table(self, state_manager: SwarmStateManager) -> None:
        """Test the DB creation.

        A DB with the swarm table must be created when missing.

        Args:
            state_manager (SwarmStateManager): State manager.
        """
        state_manager._create_db_if_missing()
        with sqlite3.connect(state_manager.db_path) as cnx:
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
        with sqlite3.connect(state_manager.db_path) as cnx:
            cursor = cnx.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='swarm';"
            )
            # one-element tuple
            name = cursor.fetchone()

        assert name[0] == "swarm"

    def test_get_flat_data_dict(self, state_manager: SwarmStateManager) -> None:
        """Check the JSON is flattened correctly.

        Args:
            state_manager (SwarmStateManager): State manager.
        """
        data = state_manager._get_flat_data_dict()
        assert data == {
            "name": state_manager.swarm.name,
            "jobid": state_manager.swarm.jobid,
            "lb_jobid": state_manager.swarm.lb_jobid,
            "lb_node": state_manager.swarm.lb_node,
            "endpoint": state_manager.swarm.endpoint,
            "delete_on_exit": state_manager.swarm.delete_on_exit,
            "model": state_manager.swarm.model,
            "driver_cpus_per_task": state_manager.swarm.cfg.driver.cpus_per_task,
            "driver_mem": state_manager.swarm.cfg.driver.mem,
            "driver_threads_per_core": state_manager.swarm.cfg.driver.threads_per_core,
            "driver_wall_time": state_manager.swarm.cfg.driver.wall_time,
            "driver_enable_proxy_buffering": state_manager.swarm.cfg.driver.enable_proxy_buffering,
            "driver_nginx_timeout": state_manager.swarm.cfg.driver.nginx_timeout,
            "hf_home": str(state_manager.swarm.cfg.hf_home),
            "revision": state_manager.swarm.cfg.revision,
            "replicas": state_manager.swarm.cfg.replicas,
            "gpus_per_replica": state_manager.swarm.cfg.gpus_per_replica,
            "gpus_per_node": state_manager.swarm.cfg.gpus_per_node,
            "replicas_per_node": state_manager.swarm.cfg.replicas_per_node,
            "nodes": state_manager.swarm.cfg.nodes,
            "cpus_per_task": state_manager.swarm.cfg.cpus_per_task,
            "requires_ray": state_manager.swarm.cfg.requires_ray,
            "mem_per_cpu": state_manager.swarm.cfg.mem_per_cpu,
            "partition": state_manager.swarm.cfg.partition,
            "account": state_manager.swarm.cfg.account,
            "vllm_image": str(state_manager.swarm.cfg.vllm_image),
            "nginx_image": str(state_manager.swarm.cfg.nginx_image),
            "lb_wait": state_manager.swarm.cfg.lb_wait,
            "lb_port": state_manager.swarm.cfg.lb_port,
            "home_directory": str(state_manager.swarm.cfg.home_directory),
            "log_directory": str(state_manager.swarm.cfg.log_directory),
            "max_concurrent_requests": state_manager.swarm.cfg.max_concurrent_requests,
            "poll_interval": state_manager.swarm.cfg.poll_interval,
            "template_path": str(state_manager.swarm.cfg.template_path),
            "nginx_template_path": str(state_manager.swarm.cfg.nginx_template_path),
            "vllm_args": state_manager.swarm.cfg.vllm_args,
            "vllm_port": state_manager.swarm.cfg.vllm_port,
            "ray_port": state_manager.swarm.cfg.ray_port,
            "ray_dashboard_port": state_manager.swarm.cfg.ray_dashboard_port,
            "venv_path": state_manager.swarm.cfg.venv_path,
            "time_limit": state_manager.swarm.cfg.time_limit,
            "exclude_nodes": state_manager.swarm.cfg.exclude_nodes,
            "node_list": state_manager.swarm.cfg.node_list,
            "mail_user": state_manager.swarm.cfg.mail_user,
        }

    def test_save(self, state_manager: SwarmStateManager) -> None:
        """Check the record is saved in the DB.

        Args:
            state_manager (SwarmStateManager): State manager.
        """
        state_manager._create_db_if_missing()
        state_manager.save()
        with sqlite3.connect(state_manager.db_path) as cnx:
            cursor = cnx.cursor()
            cursor.execute(
                f"SELECT * FROM swarm WHERE jobid = {state_manager.swarm.jobid};"
            )
            row = cursor.fetchall()

        assert len(row) == 1

    def test_update_lb_data(self, state_manager: SwarmStateManager) -> None:
        """Test LB data is updated correctly.

        Args:
            state_manager (SwarmStateManager): State manager.
        """
        state_manager._create_db_if_missing()
        state_manager.save()
        state_manager.swarm.cfg.lb_port = 7777
        state_manager.swarm.lb_node = "node2"
        state_manager.swarm.endpoint = "http://updated-lb:7777"

        state_manager.update_lb_data()

        with sqlite3.connect(state_manager.db_path) as cnx:
            cursor = cnx.cursor()
            cursor.execute(
                "SELECT lb_port, lb_node, endpoint FROM swarm"
                f" WHERE jobid = {state_manager.swarm.jobid};"
            )
            lb_port, lb_node, endpoint = cursor.fetchone()

        assert lb_port == state_manager.swarm.cfg.lb_port
        assert lb_node == state_manager.swarm.lb_node
        assert endpoint == state_manager.swarm.endpoint

    def test_load(self, state_manager: SwarmStateManager) -> None:
        """Test the swarm initialization from a record.

        Args:
            state_manager (SwarmStateManager): State manager.
        """
        state_manager._create_db_if_missing()
        state_manager.save()

        swarm = SwarmStateManager.load(
            state_manager.swarm.jobid,
            home_directory=state_manager.swarm.cfg.home_directory,
        )

        assert swarm.jobid == state_manager.swarm.jobid
        assert swarm.name == state_manager.swarm.name

    def test_load_unknown_job(self, state_manager: SwarmStateManager) -> None:
        """Exception is raised if the swarm is not found in the DB.

        Args:
            state_manager (SwarmStateManager): State manager.
        """
        with pytest.raises(JobNotFoundError):
            SwarmStateManager.load(
                state_manager.swarm.jobid,
                home_directory=state_manager.swarm.cfg.home_directory,
            )

    def test_delete_record(self, state_manager: SwarmStateManager) -> None:
        """Test the record deletion from the DB.

        Args:
            state_manager (SwarmStateManager): State manager.
        """
        state_manager._create_db_if_missing()
        state_manager.save()
        state_manager.swarm.delete_record()

        with sqlite3.connect(state_manager.db_path) as cnx:
            cursor = cnx.cursor()
            cursor.execute(
                f"SELECT * FROM swarm WHERE jobid = {state_manager.swarm.jobid};"
            )
            rows = cursor.fetchall()

        assert not rows
