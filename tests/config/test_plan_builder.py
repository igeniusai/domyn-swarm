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


def test_plan_builder_requires_backend():
    cfg = SimpleNamespace(backend=None)

    with pytest.raises(ValueError, match="At least one backend"):
        PlanBuilder(cfg).build()
