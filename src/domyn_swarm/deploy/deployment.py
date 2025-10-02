from dataclasses import dataclass
from typing import Optional

from domyn_swarm.platform.protocols import (
    ComputeBackend,
    JobHandle,
    ServingBackend,
    ServingHandle,
    ServingStatus,
)


@dataclass
class Deployment:
    """Composes a ServingBackend and a ComputeBackend.

    Typical flow:
    >>> dep = Deployment(serving=..., compute=...)
    >>> eph = dep.up("my-endpoint", serving_spec={"model": "..."}, timeout_s=1800)
    >>> env = {"ENDPOINT": eph.url, "MODEL": "mymodel"}
    >>> jh = dep.run(name="my-job", image="...", command=["python", "-m", "..."], env=env)
    >>> dep.down(eph)

    Use as a context manager to ensure cleanup of the serving endpoint
    if exceptions occur: the `ServingHandle` is stored on enter and removed on exit.
    """

    serving: ServingBackend
    compute: ComputeBackend = None  # type: ignore[assignment]

    extras: Optional[dict] = (
        None  # for any extra metadata, e.g. workspace, resource shape, etc.
    )
    _handle: Optional[ServingHandle] = None

    def up(self, name: str, serving_spec: dict, timeout_s: int) -> ServingHandle:
        """Create and wait for a serving endpoint to be ready."""
        handle = self.serving.create_or_update(name, serving_spec, extras=self.extras)
        handle = self.serving.wait_ready(handle, timeout_s, extras=self.extras)
        self._handle = handle
        return handle

    def run(
        self,
        *,
        name: str,
        image: Optional[str],
        command: list[str],
        env: dict[str, str],
        resources: Optional[dict] = None,
        detach: bool = False,
        nshards: Optional[int] = None,
        shard_id: Optional[int] = None,
    ) -> JobHandle:
        """Submit a job to the compute backend that targets the serving endpoint."""
        return self.compute.submit(
            name=name,
            image=image,
            command=command,
            env=env,
            resources=resources,
            detach=detach,
            nshards=nshards,
            shard_id=shard_id,
            extras=self.extras,
        )

    def down(self, handle: ServingHandle) -> None:
        """Delete the serving endpoint."""
        self.serving.delete(handle)

    # --- Context manager sugar ------------------------------------------------
    def __enter__(self) -> "Deployment":  # type: ignore[override]
        return self

    def __exit__(self, exc_type, exc, tb):  # type: ignore[override]
        if self._handle is not None:
            try:
                self.down(self._handle)
            finally:
                self._handle = None

    def ensure_ready(self):
        """Ensure the current serving handle is ready, or raise if not."""
        if self._handle is None:
            raise RuntimeError("No serving handle to ensure readiness for")
        return self.serving.ensure_ready(self._handle)

    def status(self) -> ServingStatus:
        """Get the current status of the deployment."""
        if self._handle is None:
            raise RuntimeError("No serving handle to get status for")
        return self.serving.status(self._handle)
