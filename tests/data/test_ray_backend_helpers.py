import types

import pytest

from domyn_swarm.data.backends.ray_backend import BackendError, RayBackend


def test_ray_backend_schema_from_dict():
    """Reads schema from dict-like object."""
    backend = RayBackend()
    data = types.SimpleNamespace(schema=lambda: {"a": "int", "b": "str"})
    assert backend.schema(data) == {"a": "int", "b": "str"}


def test_ray_backend_schema_from_pairs():
    """Reads schema from names/types sequences."""
    backend = RayBackend()
    schema = types.SimpleNamespace(names=["a"], types=["int"])
    data = types.SimpleNamespace(schema=lambda: schema)
    assert backend.schema(data) == {"a": "int"}


def test_ray_backend_write_rejects_shards(tmp_path):
    """Raises when shard count is requested."""
    backend = RayBackend()
    with pytest.raises(BackendError):
        backend.write(types.SimpleNamespace(), tmp_path, nshards=2)
