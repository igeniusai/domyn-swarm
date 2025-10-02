import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, runtime_checkable


class JobStatus(str, Enum):
    """Standardized job lifecycle states across platforms."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ServingPhase(str, Enum):
    """Standardized serving endpoint lifecycle phases across platforms."""

    UNKNOWN = "UNKNOWN"
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    FAILED = "FAILED"
    STOPPED = "STOPPED"
    INITIALIZING = "INITIALIZING"


@dataclass
class ServingStatus:
    phase: ServingPhase
    url: str | None
    detail: Optional[dict[str, Any]] = None


@dataclass
class ServingHandle:
    """Opaque handle for a serving endpoint on any platform.

    Attributes
    ----------
    id: str
        Provider-specific identifier (e.g., Azure endpoint name, Lepton endpoint ID,
        Slurm LB jobid, etc.).
    url: str
        Base URL to call (e.g., "https://.../v1"). Empty until ready.
    meta: dict[str, Any]
        Arbitrary metadata (ports, job ids, workspace, resource shape, etc.).
    """

    id: str
    url: str
    meta: dict[str, Any]


@dataclass
class JobHandle:
    """Opaque handle for a compute job that targets a serving endpoint.

    Attributes
    ----------
    id: str
        Provider-specific job identifier (e.g., Azure job name, Lepton Job ID,
        local PID for detached Slurm srun, etc.).
    status: JobStatus
        Current status, best-effort normalized.
    meta: dict[str, Any]
        Arbitrary metadata to assist monitoring/canceling.
    """

    id: str
    status: JobStatus
    meta: dict[str, Any]


@runtime_checkable
class ServingBackend(Protocol):
    """Create/update/delete a serving endpoint.

    This represents the "Create an endpoint" half of the user's target model.
    """

    def create_or_update(
        self, name: str, spec: dict, extras: dict | None = None
    ) -> ServingHandle: ...

    def wait_ready(
        self, handle: ServingHandle, timeout_s: int, extras: dict | None = None
    ) -> ServingHandle: ...

    def delete(self, handle: ServingHandle) -> None: ...

    def ensure_ready(self, handle: ServingHandle): ...

    def status(self, handle: ServingHandle) -> ServingStatus: ...


@runtime_checkable
class ComputeBackend(Protocol):
    """Run a container/command targeting a given endpoint.

    This represents the "Submit a job which targets said endpoint" half.
    """

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
    ) -> JobHandle: ...

    def wait(self, handle: JobHandle, *, stream_logs: bool = True) -> JobStatus: ...

    def cancel(self, handle: JobHandle) -> None: ...

    def default_python(self, cfg) -> str: ...

    def default_image(self, cfg) -> Optional[str]: ...

    def default_resources(self, cfg) -> Optional[dict]: ...

    def default_env(self, cfg) -> Dict[str, str]: ...


class DefaultComputeMixin:
    def default_python(self, cfg) -> str:
        return sys.executable

    def default_image(self, cfg) -> Optional[str]:
        return None

    def default_resources(self, cfg) -> Optional[dict]:
        return None

    def default_env(self, cfg) -> Dict[str, str]:
        return {}
