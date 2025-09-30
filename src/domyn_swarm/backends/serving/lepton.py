import logging
from dataclasses import dataclass
from typing import Optional

from domyn_swarm.config.settings import get_settings
from domyn_swarm.helpers.lepton import (
    get_env_var_by_name,
    sanitize_tokens_in_deployment,
)
from domyn_swarm.helpers.logger import setup_logger
from domyn_swarm.platform.protocols import ServingBackend, ServingHandle
from domyn_swarm.utils.imports import _require_lepton, make_lepton_client

logger = setup_logger(__name__, level=logging.INFO)

settings = get_settings()


@dataclass
class LeptonServingBackend(ServingBackend):  # type: ignore[misc]
    """DGX Cloud Lepton endpoint backend via leptonai Python SDK.

    Docs (Aug 2025):
    - Python SDK intro & examples
      https://docs.nvidia.com/dgx-cloud/lepton/reference/api/
    - Endpoint configs
      https://docs.nvidia.com/dgx-cloud/lepton/features/endpoint/configurations/

    Auth
    ----
    Run `pip install -U leptonai` and `lep login` once to store credentials.

    spec keys (suggested)
    ---------------------
    {
      "name": str,                         # endpoint name
      "image": str | None,                # container image or NIM preset
      "env": dict[str,str],               # environment variables
      "replicas": int,                    # initial replicas
      "resource_shape": str,              # Lepton resource shape name
      "token_protected": bool,            # whether endpoint requires token
      "public": bool,                     # whether endpoint is public
    }
    """

    workspace: Optional[str] = None  # if multiple workspaces, else default

    _client_cached = None
    workspace: Optional[str] = None  # if multiple workspaces, else default

    def _client(self):
        if self._client_cached is None:
            _require_lepton()  # quick availability check
            token = (
                settings.lepton_api_token.get_secret_value()
                if settings.lepton_api_token
                else None
            )
            self._client_cached = make_lepton_client(
                token=token,
                workspace=getattr(self, "workspace", None),
            )
        return self._client_cached

    def create_or_update(
        self, name: str, spec: dict, extras: dict | None = None
    ) -> ServingHandle:
        """
        Create or update a Lepton deployment (serving endpoint).
        If the deployment already exists, it will be updated with the new spec.
        """
        _require_lepton()
        from leptonai.api.v1.types.common import Metadata, SecretItem
        from leptonai.api.v1.types.deployment import (
            LeptonDeployment,
            LeptonDeploymentUserSpec,
        )

        client = self._client()

        lepton_dep_user_spec = LeptonDeploymentUserSpec.model_validate(
            spec, by_alias=True
        )

        dep = LeptonDeployment(
            metadata=Metadata(name=name),
            spec=lepton_dep_user_spec,
        )

        request_success = client.deployment.create(dep)  # type: ignore
        if not request_success:
            raise RuntimeError(f"Failed to create Lepton deployment {name}")
        deployed: LeptonDeployment = client.deployment.get(name)

        url = (
            deployed.status.endpoint.external_endpoint
            if deployed.status and deployed.status.endpoint
            else ""
        )
        logger.info(f"Lepton deployment {name} created with URL: {url}")

        token = (
            dep.spec.api_tokens[0].value
            if dep.spec and dep.spec.api_tokens and len(dep.spec.api_tokens) > 0
            else None
        )
        secret_name = (
            get_env_var_by_name(dep.spec.envs, "API_TOKEN_SECRET_NAME")
            if dep.spec and dep.spec.envs
            else None
        )
        secrets = [
            SecretItem(name=secret_name or f"{name}-token", value=token or "changeme")
        ]
        _ = client.secret.create(secrets)

        return ServingHandle(
            id=name,
            url=url,
            meta={
                "raw": sanitize_tokens_in_deployment(deployed),
                "name": name,
                "token_secret_name": secret_name,
            },
        )

    def wait_ready(
        self, handle: ServingHandle, timeout_s: int, extras: dict | None = None
    ) -> ServingHandle:
        _require_lepton()
        import time

        from leptonai.api.v1.types.deployment import (
            LeptonDeployment,
            LeptonDeploymentState,
            LeptonDeploymentStatus,
        )

        client = self._client()
        start = time.time()
        while True:
            dep: LeptonDeployment = client.deployment.get(handle.meta["name"])
            status: Optional[LeptonDeploymentStatus] = dep.status
            state = status.state if status else None
            if state == LeptonDeploymentState.Ready and status and status.endpoint:
                break
            if time.time() - start > timeout_s:
                raise TimeoutError(
                    f"Timed out waiting for Lepton deployment {handle.id} to be READY"
                )
            time.sleep(5)

        return handle

    def delete(self, handle: ServingHandle) -> None:
        client = self._client()
        try:
            client.deployment.delete(handle.id)
            secret_name = handle.meta.get("token_secret_name")
            if secret_name:
                client.secret.delete(secret_name)
        except Exception:
            pass

    def ensure_ready(self, handle: ServingHandle):
        """Ensure the current serving handle is ready, or raise if not."""
        endpoint = handle.url
        if not endpoint:
            raise RuntimeError(f"Swarm not ready (endpoint): {endpoint}")
