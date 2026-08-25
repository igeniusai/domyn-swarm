# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from domyn_swarm.data import BackendError, get_backend
from domyn_swarm.data.backends.base import DataBackend


def _get_backend(backend_name: str) -> DataBackend:
    """Resolve a data backend by name.

    Args:
        backend_name: Backend name to load.

    Returns:
        Loaded DataBackend instance.

    Raises:
        RuntimeError: If the backend cannot be resolved.
    """
    try:
        return get_backend(backend_name)
    except BackendError as exc:
        raise RuntimeError(str(exc)) from exc
