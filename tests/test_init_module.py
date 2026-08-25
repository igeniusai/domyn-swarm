# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import importlib
import warnings

import pytest

import domyn_swarm


def test_init_getattr_exposes_classes():
    """Provides lazy attribute access for public classes."""
    assert domyn_swarm.DomynLLMSwarm.__name__ == "DomynLLMSwarm"
    assert domyn_swarm.DomynLLMSwarmConfig.__name__ == "DomynLLMSwarmConfig"
    assert domyn_swarm.SwarmJob.__name__ == "SwarmJob"


@pytest.mark.parametrize("name", domyn_swarm.__all__)
def test_every_public_name_resolves(name):
    """Every name advertised in ``__all__`` is importable from the package root."""
    assert getattr(domyn_swarm, name) is not None


def test_unknown_attribute_still_raises():
    """Names outside ``__all__`` keep raising AttributeError."""
    with pytest.raises(AttributeError):
        getattr(domyn_swarm, "definitely_not_exported")  # noqa: B009


def test_public_names_import_without_deprecation_warnings():
    """Resolving the documented public surface emits no DeprecationWarning."""
    importlib.reload(domyn_swarm)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for name in domyn_swarm.__all__:
            getattr(domyn_swarm, name)
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert not deprecations, [str(w.message) for w in deprecations]
