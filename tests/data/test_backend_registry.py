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

import pytest

from domyn_swarm.data.backends import BackendError, get_backend
import domyn_swarm.data.backends.registry as registry


def test_get_backend_defaults_to_pandas():
    backend = get_backend(None)
    assert backend.name == "pandas"


def test_get_backend_rejects_unknown():
    with pytest.raises(BackendError, match="Unknown data backend"):
        get_backend("nope")


def test_get_backend_polars_missing(monkeypatch):
    monkeypatch.setattr(
        registry, "_require_polars", lambda: (_ for _ in ()).throw(BackendError("missing"))
    )
    with pytest.raises(BackendError, match="missing"):
        registry.get_backend("polars")


def test_get_backend_ray_missing(monkeypatch):
    monkeypatch.setattr(
        registry, "_require_ray", lambda: (_ for _ in ()).throw(BackendError("missing"))
    )
    with pytest.raises(BackendError, match="missing"):
        registry.get_backend("ray")
