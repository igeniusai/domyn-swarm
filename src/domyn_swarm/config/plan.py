from dataclasses import dataclass
from typing import Literal

from domyn_swarm.platform.protocols import ComputeBackend, ServingBackend


@dataclass
class DeploymentPlan:
    """
    Fully constructed pair (serving, compute) plus per-backend specs
    that the app will use when calling Deployment.up()/run().
    """

    name_hint: str  # optional suffix/prefix for generated names
    serving: ServingBackend
    compute: ComputeBackend
    serving_spec: dict  # pass to Deployment.up(name, serving_spec, timeout_s)
    job_resources: dict  # optional: pass/merge when running jobs
    extras: dict  # any useful extras (e.g., workspace id)
    platform: Literal["lepton", "slurm"] = "slurm"
