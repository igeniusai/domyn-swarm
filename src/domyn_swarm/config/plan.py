# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from domyn_swarm.platform.protocols import ComputeBackend, ServingBackend, ServingHandle


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
    """A fully constructed (serving, compute) pair plus per-backend specs.

    Attributes:
        compute: The compute backend, when it can be built without a live
            serving endpoint. ``None`` for platforms whose compute backend
            depends on the handle (Slurm needs ``lb_jobid``/``lb_node``).
        compute_factory: Builds the compute backend from a ready
            :class:`ServingHandle`. Takes precedence over ``compute``.
        platform: Platform identifier, matching ``cfg.backend.type``.
        handle_validator: Checks that a handle carries everything this
            platform's backends need. ``None`` means the platform imposes no
            requirements beyond what the backends themselves check.
    """

    name_hint: str
    serving: ServingBackend
    compute: ComputeBackend | None
    serving_spec: dict
    job_resources: dict
    extras: dict
    shared_env: dict[str, str] = field(default_factory=dict)
    image: str | None = None
    timeout_s: int | None = None
    platform: str = "slurm"
    compute_factory: Callable[[ServingHandle], ComputeBackend] | None = None
    handle_validator: Callable[[ServingHandle], None] | None = None

    def validate_serving_handle(self, handle: ServingHandle) -> None:
        """Check that a serving handle is usable by this platform's backends.

        Called when adopting a handle that was not produced in this process --
        typically one rehydrated from the state DB -- so an incomplete record
        fails with a clear message instead of a stray ``KeyError`` from inside
        a backend later on. Platforms without extra requirements supply no
        validator and everything passes.

        Args:
            handle: The handle about to be adopted.

        Raises:
            ValueError: If the handle is missing metadata this platform needs.
        """
        if self.handle_validator is not None:
            self.handle_validator(handle)

    def make_compute_backend(self, handle: ServingHandle) -> ComputeBackend:
        """Return the compute backend for a ready serving endpoint.

        Args:
            handle: The handle returned once the serving endpoint is ready.

        Returns:
            The compute backend to submit jobs against this endpoint.

        Raises:
            RuntimeError: If the platform supplies neither a factory nor a
                plan-time compute backend.
        """
        if self.compute_factory is not None:
            return self.compute_factory(handle)
        if self.compute is not None:
            return self.compute
        raise RuntimeError(
            f"Platform {self.platform!r} supplies no compute backend: its build() "
            "must set either `compute` or `compute_factory`."
        )


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
