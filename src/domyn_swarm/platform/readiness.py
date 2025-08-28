from typing import Protocol

from domyn_swarm.platform.protocols import ServingHandle


class ServingReadiness(Protocol):
    def wait_ready(self, handle: ServingHandle, timeout_s: int) -> ServingHandle:
        """Block (or poll) until the serving endpoint is ready and set handle.url."""
        ...
