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
