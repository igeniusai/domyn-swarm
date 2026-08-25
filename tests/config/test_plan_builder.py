# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from domyn_swarm.config.plan import DeploymentPlan, PlanBuilder


class BackendStub:
    def __init__(self, plan):
        self._plan = plan

    def build(self, cfg_ctx):
        return self._plan


def test_plan_builder_normalizes_missing_fields():
    plan = DeploymentPlan(
        name_hint="stub",
        serving=object(),
        compute=object(),
        serving_spec=None,  # type: ignore[arg-type]
        job_resources=None,  # type: ignore[arg-type]
        extras=None,  # type: ignore[arg-type]
        shared_env=None,  # type: ignore[arg-type]
        platform="slurm",
    )
    cfg = SimpleNamespace(backend=BackendStub(plan))

    out = PlanBuilder(cfg).build()

    assert out.serving_spec == {}
    assert out.job_resources == {}
    assert out.extras == {}
    assert out.shared_env == {}


def test_plan_builder_merges_cfg_env():
    plan = DeploymentPlan(
        name_hint="stub",
        serving=object(),
        compute=object(),
        serving_spec={},
        job_resources={},
        extras={},
        shared_env={"A": "B"},
        platform="slurm",
    )
    cfg = SimpleNamespace(backend=BackendStub(plan), env={"C": "D"})

    out = PlanBuilder(cfg).build()

    assert out.shared_env == {"A": "B", "C": "D"}


def test_plan_builder_defaults_image_from_backend_job():
    plan = DeploymentPlan(
        name_hint="stub",
        serving=object(),
        compute=object(),
        serving_spec={},
        job_resources={},
        extras={},
        platform="slurm",
    )
    backend = BackendStub(plan)
    backend.job = SimpleNamespace(image="repo/image:tag")
    cfg = SimpleNamespace(backend=backend)

    out = PlanBuilder(cfg).build()

    assert out.image == "repo/image:tag"


def test_plan_builder_preserves_job_resources():
    plan = DeploymentPlan(
        name_hint="stub",
        serving=object(),
        compute=object(),
        serving_spec={},
        job_resources={"cpu": 4},
        extras={},
        platform="slurm",
    )
    cfg = SimpleNamespace(backend=BackendStub(plan))

    out = PlanBuilder(cfg).build()

    assert out.job_resources == {"cpu": 4}


def test_plan_builder_requires_backend():
    cfg = SimpleNamespace(backend=None)

    with pytest.raises(ValueError, match="At least one backend"):
        PlanBuilder(cfg).build()
