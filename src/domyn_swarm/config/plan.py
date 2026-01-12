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

from dataclasses import dataclass, field
from typing import Any, Literal

from domyn_swarm.platform.protocols import ComputeBackend, ServingBackend


@dataclass
class DeploymentContext:
    """Normalized deployment context shared across serving + compute."""

    serving_spec: dict = field(default_factory=dict)
    job_resources: dict | None = None
    extras: dict = field(default_factory=dict)
    timeout_s: int | None = None
    shared_env: dict[str, str] = field(default_factory=dict)
    image: str | None = None


@dataclass
class DeploymentPlan:
    """
    Fully constructed pair (serving, compute) plus per-backend specs
    that the app will use when calling Deployment.up()/run().
    """

    name_hint: str  # optional suffix/prefix for generated names
    serving: ServingBackend
    compute: ComputeBackend
    serving_spec: dict  # pass to DeploymentContext for Deployment.up(...)
    job_resources: dict  # optional: pass/merge when running jobs
    extras: dict  # any useful extras (e.g., workspace id)
    shared_env: dict[str, str] = field(default_factory=dict)
    image: str | None = None
    timeout_s: int | None = None
    platform: Literal["lepton", "slurm"] = "slurm"


class PlanBuilder:
    """Plan assembly entry point that normalizes plan fields."""

    def __init__(self, cfg_ctx: Any):
        self.cfg_ctx = cfg_ctx

    def build(self) -> DeploymentPlan:
        backend = getattr(self.cfg_ctx, "backend", None)
        if backend is None:
            raise ValueError("At least one backend must be configured")
        plan = backend.build(self.cfg_ctx)
        return self._normalize(plan)

    def _normalize(self, plan: DeploymentPlan) -> DeploymentPlan:
        plan = self._normalize_static(plan)
        cfg_env = getattr(self.cfg_ctx, "env", None)
        if cfg_env:
            plan.shared_env.update(cfg_env)
        if plan.image is None:
            plan.image = self._default_job_image()
        return plan

    @staticmethod
    def _normalize_static(plan: DeploymentPlan) -> DeploymentPlan:
        if plan.serving_spec is None:
            plan.serving_spec = {}
        if plan.job_resources is None:
            plan.job_resources = {}
        if plan.extras is None:
            plan.extras = {}
        if plan.shared_env is None:
            plan.shared_env = {}
        return plan

    def _default_job_image(self) -> str | None:
        backend = getattr(self.cfg_ctx, "backend", None)
        job_cfg = getattr(backend, "job", None)
        return getattr(job_cfg, "image", None)
