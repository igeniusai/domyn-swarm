from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

from domyn_swarm.config.lepton import LeptonConfig
from domyn_swarm.platform.protocols import DefaultComputeMixin, JobHandle, JobStatus
from domyn_swarm.utils.imports import _require_lepton


@dataclass
class LeptonComputeBackend(DefaultComputeMixin):  # type: ignore[misc]
    """DGX Cloud Lepton batch job backend via leptonai Python SDK.

    Docs (Aug 2025): https://docs.nvidia.com/dgx-cloud/lepton/reference/api/

    Submit a container that runs your runner CLI. You can pass endpoint URL and model
    via environment variables; secrets can be injected via Lepton workspace secrets.

    resources (suggested)
    ---------------------
    {
      "resource_shape": str,              # e.g., A100x1_80GB
      "node_group": str | None,          # optional: constrain to a node group
      "allowed_nodes": list[str] | None, # optional: subset of nodes
      "completions": int,                # default 1
      "parallelism": int,                # default 1
    }
    """

    workspace: Optional[str] = None  # if multiple workspaces, else default

    def _client(self):
        _require_lepton()
        try:
            from leptonai.api.v2.client import APIClient
        except Exception as e:
            raise ImportError(
                "Install leptonai and run `lep login` to use Lepton backends"
            ) from e
        return APIClient()

    def submit(
        self,
        *,
        name: str,
        image: Optional[str],
        command: Sequence[str],
        env: Optional[Mapping[str, str]] = None,
        resources: Optional[dict] = None,
        detach: bool = False,
        nshards: Optional[int] = None,
        shard_id: Optional[int] = None,
        extras: dict | None = None,
    ) -> JobHandle:
        _require_lepton()
        from leptonai.api.v1.types.common import Metadata
        from leptonai.api.v1.types.deployment import (
            EnvValue,
            EnvVar,
            LeptonContainer,
        )
        from leptonai.api.v1.types.job import (
            LeptonJob,
            LeptonJobUserSpec,
        )

        client = self._client()

        container = LeptonContainer(
            image=image,
            command=[*map(str, command)],
        )

        spec = LeptonJobUserSpec.model_validate(resources or {})
        spec.container = container
        secret_name = extras.get("api_token") if extras else None

        if spec.envs is None:
            spec.envs = []
        spec.envs.append(
            EnvVar(name="API_TOKEN", value_from=EnvValue(secret_name_ref=secret_name))
        )

        job = LeptonJob(spec=spec, metadata=Metadata(name=name))

        created = client.job.create(job)
        print(dir(created))
        job_id = created.metadata.id_ if created and created.metadata else None
        if not job_id:
            raise RuntimeError("Failed to create Lepton job")
        return JobHandle(id=job_id, status=JobStatus.PENDING, meta={"raw": created})

    def wait(self, handle: JobHandle, *, stream_logs: bool = True) -> JobStatus:
        _require_lepton()
        from leptonai.api.v1.types.deployment import (
            LeptonDeploymentState,
        )
        from leptonai.api.v1.types.job import (
            LeptonJobState,
        )

        client = self._client()

        job = client.job.get(handle.id)
        state: Optional[LeptonJobState] = job.status.state if job.status else None
        if state in {LeptonDeploymentState.Stopped, LeptonDeploymentState.Stopping}:
            return JobStatus.FAILED
        if state == LeptonDeploymentState.Ready:
            return JobStatus.RUNNING
        return JobStatus.SUCCEEDED

    def cancel(self, handle: JobHandle) -> None:
        client = self._client()
        try:
            client.job.delete(handle.id)
        except Exception:
            pass

    def default_python(self, cfg) -> str:
        return "python"

    def default_image(self, cfg: LeptonConfig) -> Optional[str]:
        # if you populated cfg.lepton.job.image, reuse it
        return cfg.job.image if getattr(cfg, "job", None) else None

    def default_resources(self, cfg: LeptonConfig) -> Optional[dict]:
        _require_lepton()
        from leptonai.api.v1.types.job import (
            LeptonJobUserSpec,
            LeptonResourceAffinity,
        )

        affinity = LeptonResourceAffinity(
            allowed_dedicated_node_groups=cfg.job.allowed_dedicated_node_groups,
            allowed_nodes_in_node_group=cfg.job.allowed_nodes or None,
        )
        spec = LeptonJobUserSpec(
            affinity=affinity,
            resource_shape=cfg.job.resource_shape or cfg.endpoint.resource_shape,
            completions=1,
            parallelism=1,
            mounts=cfg.job.mounts,
        )
        return spec.model_dump(by_alias=True)

    def default_env(self, cfg) -> Dict[str, str]:
        # forward secret name for the endpoint token if you stored it in the handle
        return {}
