# Copyright 2025 Domyn
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
