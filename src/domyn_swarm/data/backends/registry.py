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

from __future__ import annotations

from domyn_swarm.data.backends.base import BackendError, DataBackend
from domyn_swarm.data.backends.pandas_backend import PandasBackend


def get_backend(name: str | None) -> DataBackend:
    backend = (name or "pandas").strip().lower()
    if backend in {"pandas", "pd"}:
        return PandasBackend()
    if backend == "polars":
        _require_polars()
        from domyn_swarm.data.backends.polars_backend import PolarsBackend

        return PolarsBackend()
    if backend == "ray":
        _require_ray()
        from domyn_swarm.data.backends.ray_backend import RayBackend

        return RayBackend()
    raise BackendError(f"Unknown data backend: {backend}")


def _require_polars() -> None:
    try:
        import polars  # noqa: F401
    except Exception as exc:
        raise BackendError("Polars backend requires `polars` to be installed.") from exc


def _require_ray() -> None:
    try:
        import ray.data  # noqa: F401
    except Exception as exc:
        raise BackendError("Ray backend requires `ray[data]` to be installed.") from exc
