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

from __future__ import annotations

from typing import TYPE_CHECKING

from domyn_swarm.utils.imports import _require_lepton

if TYPE_CHECKING:
    from leptonai.api.v1.types.deployment import EnvVar, LeptonDeployment


def get_env_var_by_name(env_vars: list[EnvVar], name: str) -> str | None:
    _require_lepton()
    """Get the value of an environment variable by name from a list of EnvVar."""
    for env_var in env_vars:
        if env_var.name == name:
            return env_var.value
    return None


def sanitize_tokens_in_deployment(dep: LeptonDeployment | dict) -> LeptonDeployment:
    """Redact any API tokens in a LeptonDeployment for safe logging."""
    _require_lepton()
    from leptonai.api.v1.types.deployment import LeptonDeployment

    if isinstance(dep, dict):
        dep = LeptonDeployment.model_validate(dep)
    if dep.spec and dep.spec.api_tokens:
        for token_var in dep.spec.api_tokens:
            token_var.value = "REDACTED"

    return dep
