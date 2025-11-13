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

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING

import requests
from requests import RequestException

from domyn_swarm.config.settings import get_settings
from domyn_swarm.helpers.lepton import (
    get_env_var_by_name,
    sanitize_tokens_in_deployment,
)
from domyn_swarm.helpers.logger import setup_logger
from domyn_swarm.platform.protocols import (
    ServingBackend,
    ServingHandle,
    ServingPhase,
    ServingStatus,
)
from domyn_swarm.utils.imports import _require_lepton, make_lepton_client

if TYPE_CHECKING:
    from leptonai.api.v2.client import APIClient

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

    workspace: str | None = None  # if multiple workspaces, else default

    _client_cached = None
    workspace: str | None = None  # if multiple workspaces, else default

    def _client(self) -> "APIClient":
        if self._client_cached is None:
            _require_lepton()  # quick availability check
            token = (
                settings.lepton_api_token.get_secret_value() if settings.lepton_api_token else None
            )
            self._client_cached = make_lepton_client(
                token=token,
                workspace=getattr(self, "workspace", None),
            )
        return self._client_cached

    def create_or_update(self, name: str, spec: dict, extras: dict) -> ServingHandle:
        """
        Create or update a Lepton deployment (serving endpoint).
        If the deployment already exists, it will be updated with the new spec.
        """
        _require_lepton()
        from leptonai.api.v1.types.common import Metadata
        from leptonai.api.v1.types.deployment import (
            LeptonDeployment,
            LeptonDeploymentUserSpec,
        )
        from leptonai.api.v1.types.secret import SecretItem

        client = self._client()

        lepton_dep_user_spec = LeptonDeploymentUserSpec.model_validate(spec, by_alias=True)

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
        logger.info(f"Lepton deployment [cyan]{name}[/cyan] created with URL: {url}")

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
        secret_name = secret_name or f"{name}-token"
        secrets = [SecretItem(name=secret_name, value=token or "changeme")]
        _ = client.secret.create(secrets)

        raw = sanitize_tokens_in_deployment(deployed)
        return ServingHandle(
            id=name,
            url=url,
            meta={
                "raw": raw.model_dump(),
                "name": name,
                "token_secret_name": secret_name,
            },
        )

    def wait_ready(self, handle: ServingHandle, timeout_s: int, extras: dict) -> ServingHandle:
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
            status: LeptonDeploymentStatus | None = dep.status
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

    def status(self, handle: ServingHandle) -> ServingStatus:
        """
        One-shot status:
        - Ask Lepton for the deployment state.
        - If state looks 'ready', do a quick HTTP probe to ensure the endpoint is actually up.
        - Map to ServingPhase and return a ServingStatus with details.
        """
        from leptonai.api.v1.types.deployment import (
            LeptonDeployment,
            LeptonDeploymentState,
        )

        try:
            client = self._client()
        except Exception as e:
            return ServingStatus(
                phase=ServingPhase.UNKNOWN,
                url=handle.url,
                detail={"reason": "lepton_client_failed", "error": str(e)},
            )

        name = handle.meta.get("name", handle.id)

        try:
            dep: LeptonDeployment = client.deployment.get(name)
        except Exception as e:
            # Network/API issue - report unknown with context
            return ServingStatus(
                phase=ServingPhase.UNKNOWN,
                url=handle.url,
                detail={"reason": "lepton_get_failed", "error": str(e)},
            )

        state: LeptonDeploymentState | None = dep.status.state if dep and dep.status else None
        url = (
            dep.status.endpoint.external_endpoint
            if dep and dep.status and dep.status.endpoint
            else handle.url
        )

        # Map Lepton state to coarse phase first (scheduler view)
        if state in {
            LeptonDeploymentState.Stopped,
            getattr(LeptonDeploymentState, "Error", None),
        }:
            return ServingStatus(
                phase=ServingPhase.FAILED,
                url=url,
                detail={"raw_state": getattr(state, "value", str(state))},
            )
        if state is None:
            return ServingStatus(phase=ServingPhase.UNKNOWN, url=url, detail={"raw_state": None})
        if state != LeptonDeploymentState.Ready:
            # Deploying / Scaling / Updating etc.
            return ServingStatus(
                phase=ServingPhase.PENDING,
                url=url,
                detail={"raw_state": getattr(state, "value", str(state))},
            )

        # State says Ready — verify HTTP is answering to avoid false positives
        http_ok = False
        http_code: int | None = None
        if url:
            try:
                r = requests.get(f"{url.rstrip('/')}/health", timeout=1.5)
                http_code = r.status_code
                http_ok = r.status_code == 200
            except RequestException:
                http_ok = False

        if http_ok:
            # Cache the discovered URL back on the handle
            handle.url = url
            return ServingStatus(
                phase=ServingPhase.RUNNING,
                url=url,
                detail={
                    "raw_state": LeptonDeploymentState.Ready.value,
                    "http": http_code,
                },
            )

        # Lepton says Ready but HTTP not yet responding - treat as initializing
        return ServingStatus(
            phase=ServingPhase.INITIALIZING,
            url=url,
            detail={
                "raw_state": LeptonDeploymentState.Ready.value,
                "http": http_code or "unreachable",
            },
        )
