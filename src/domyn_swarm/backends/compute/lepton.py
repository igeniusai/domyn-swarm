from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from leptonai.api.v1.types.common import Metadata
from leptonai.api.v1.types.deployment import LeptonContainer, LeptonDeploymentState
from leptonai.api.v1.types.job import (
    EnvVar,
    LeptonJob,
    LeptonJobState,
    LeptonJobUserSpec,
)

from domyn_swarm.platform.protocols import ComputeBackend, JobHandle, JobStatus


@dataclass
class LeptonComputeBackend(ComputeBackend):  # type: ignore[misc]
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

    def _client(self):
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
    ) -> JobHandle:
        client = self._client()
        res = resources or {}

        container = LeptonContainer(
            image=image,
            command=[*map(str, command)],
        )

        affin = None
        ng = res.get("node_group")
        allowed_nodes = res.get("allowed_nodes")
        if ng or allowed_nodes:
            from leptonai.api.v1.types.affinity import LeptonResourceAffinity

            affin = LeptonResourceAffinity(
                allowed_dedicated_node_groups=[ng] if ng else None,
                allowed_nodes_in_node_group=allowed_nodes if allowed_nodes else None,
            )

        spec = LeptonJobUserSpec(
            resource_shape=res.get("resource_shape"),
            affinity=affin,
            container=container,
            completions=res.get("completions", 1),
            parallelism=res.get("parallelism", 1),
            envs=[EnvVar(name=k, value=v) for k, v in (env or {}).items()],
        )
        job = LeptonJob(spec=spec, metadata=Metadata(name=name))
        created = client.job.create(job)
        job_id = created.metadata.id_ if created and created.metadata else None
        if not job_id:
            raise RuntimeError("Failed to create Lepton job")
        return JobHandle(id=job_id, status=JobStatus.PENDING, meta={"raw": created})

    def wait(self, handle: JobHandle, *, stream_logs: bool = True) -> JobStatus:
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
