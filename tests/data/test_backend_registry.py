# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

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
