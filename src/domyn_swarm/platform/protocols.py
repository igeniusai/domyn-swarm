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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import sys
from typing import Any, Protocol, runtime_checkable


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
    detail: dict[str, Any] | None = None


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

    def create_or_update(self, name: str, spec: dict, extras: dict) -> ServingHandle: ...

    def wait_ready(self, handle: ServingHandle, timeout_s: int, extras: dict) -> ServingHandle: ...

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
        image: str | None,
        command: Sequence[str],
        env: Mapping[str, str] | None = None,
        resources: dict | None = None,
        detach: bool = False,
        nshards: int | None = None,
        shard_id: int | None = None,
        extras: dict | None = None,
    ) -> JobHandle: ...

    def wait(self, handle: JobHandle, *, stream_logs: bool = True) -> JobStatus: ...

    def cancel(self, handle: JobHandle) -> None: ...

    def default_python(self, cfg) -> str: ...

    def default_image(self, cfg) -> str | None: ...

    def default_resources(self, cfg) -> dict | None: ...

    def default_env(self, cfg) -> dict[str, str]: ...


class DefaultComputeMixin:
    def default_python(self, cfg) -> str:
        return sys.executable

    def default_image(self, cfg) -> str | None:
        return None

    def default_resources(self, cfg) -> dict | None:
        return None

    def default_env(self, cfg) -> dict[str, str]:
        return {}
