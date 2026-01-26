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
