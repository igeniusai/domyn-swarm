# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""A DeploymentPlan must be able to finish building its own compute backend."""

import pytest

from domyn_swarm.config.slurm import SlurmConfig, SlurmEndpointConfig
from domyn_swarm.config.swarm import DomynLLMSwarmConfig
from domyn_swarm.platform.protocols import ComputeBackend, ServingHandle


def _slurm_cfg() -> DomynLLMSwarmConfig:
    return DomynLLMSwarmConfig(
        name="fake",
        hf_home=".",
        image="image",
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
            partition="partition",
            account="account",
            qos="qos",
            requires_ray=False,
            endpoint=SlurmEndpointConfig(
                cpus_per_task=1,
                nginx_image="/path/to/vllm.sif",
                mem="1GB",
                threads_per_core=1,
                wall_time="24:00:00",
                enable_proxy_buffering=True,
                nginx_timeout="60s",
            ),
        ),
    )


def _ready_slurm_handle() -> ServingHandle:
    return ServingHandle(
        id="1234",
        url="http://lb:9003",
        meta={"lb_jobid": 1234, "jobid": 5678, "lb_node": "lrdn4759", "port": 9003},
    )


def test_slurm_plan_builds_a_real_compute_backend_from_the_handle() -> None:
    """The plan, not the swarm, turns a ready handle into a compute backend."""
    plan = _slurm_cfg().build_plan()

    backend = plan.make_compute_backend(_ready_slurm_handle())

    assert isinstance(backend, ComputeBackend)
    assert backend.lb_jobid == 1234
    assert backend.lb_node == "lrdn4759"


def test_slurm_plan_holds_no_placeholder_compute_backend() -> None:
    """Before a handle exists there is no compute backend, not a fake one."""
    plan = _slurm_cfg().build_plan()

    assert plan.compute is None


def test_slurm_plan_rejects_a_handle_without_lb_metadata() -> None:
    """A handle missing lb_jobid/lb_node fails loudly, naming what is missing."""
    plan = _slurm_cfg().build_plan()
    incomplete = ServingHandle(id="1234", url="", meta={"jobid": 5678})

    with pytest.raises(RuntimeError, match="lb_jobid"):
        plan.make_compute_backend(incomplete)


def test_plan_platform_is_an_open_string() -> None:
    """`platform` must not be a closed union, or a third backend cannot be added."""
    import typing

    from domyn_swarm.config.plan import DeploymentPlan

    hints = typing.get_type_hints(DeploymentPlan)

    assert hints["platform"] is str


def test_slurm_plan_rejects_a_persisted_handle_without_a_jobid() -> None:
    """Validating a rehydrated handle is the plan's job, not the swarm's.

    ``SlurmServingBackend.status()`` subscripts ``meta["jobid"]``, so a record
    without one has to be rejected before it is adopted.
    """
    plan = _slurm_cfg().build_plan()
    no_jobid = ServingHandle(
        id="1234", url="http://lb:9003", meta={"lb_jobid": 1234, "lb_node": "n1"}
    )

    with pytest.raises(ValueError, match="job IDs"):
        plan.validate_serving_handle(no_jobid)


def test_plan_validation_accepts_a_ready_slurm_handle() -> None:
    """A complete handle passes validation without raising."""
    plan = _slurm_cfg().build_plan()

    plan.validate_serving_handle(_ready_slurm_handle())
