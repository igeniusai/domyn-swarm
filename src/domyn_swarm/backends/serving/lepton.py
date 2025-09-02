from dataclasses import dataclass
from typing import Optional

from leptonai.api.v1.types.common import Metadata
from leptonai.api.v1.types.deployment import (
    LeptonDeployment,
    LeptonDeploymentState,
    LeptonDeploymentStatus,
    LeptonDeploymentUserSpec,
)

from domyn_swarm.platform.protocols import ServingBackend, ServingHandle


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

    def _client(self):
        try:
            from leptonai.api.v2.client import APIClient
        except Exception as e:
            raise ImportError(
                "Install leptonai and run `lep login` to use Lepton backends"
            ) from e
        return APIClient()

    def create_or_update(self, name: str, spec: dict) -> ServingHandle:
        """
        Create or update a Lepton deployment (serving endpoint).
        If the deployment already exists, it will be updated with the new spec.
        """
        client = self._client()

        dep = LeptonDeployment(
            metadata=Metadata(name=name),
            spec=LeptonDeploymentUserSpec.model_validate(spec),
        )

        request_success = client.deployment.create(dep)

        if not request_success:
            raise RuntimeError(f"Failed to create Lepton deployment {name}")
        deployed: LeptonDeployment = client.deployment.get(name)
        url = (
            deployed.status.endpoint.internal_endpoint
            if deployed.status and deployed.status.endpoint
            else ""
        )
        return ServingHandle(id=name, url=url, meta={"raw": deployed, "name": name})

    def wait_ready(self, handle: ServingHandle, timeout_s: int) -> ServingHandle:
        import time

        client = self._client()
        start = time.time()
        while True:
            dep: LeptonDeployment = client.deployment.get(handle.meta["name"])
            status: Optional[LeptonDeploymentStatus] = dep.status
            state = status.state if status else None
            if state == LeptonDeploymentState.Ready and status and status.endpoint:
                handle.url = status.endpoint.internal_endpoint
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
        except Exception:
            pass
