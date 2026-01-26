import types

import pytest

from domyn_swarm.data.backends.base import BackendError
from domyn_swarm.jobs.io import backend as backend_mod, columns as columns_mod


def test_get_backend_wraps_backend_error(monkeypatch):
    """Wraps BackendError as RuntimeError with message passthrough.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """

    def _boom(_: str):
        """Raise a BackendError for testing."""
        raise BackendError("nope")

    monkeypatch.setattr(backend_mod, "get_backend", _boom)
    with pytest.raises(RuntimeError, match="nope"):
        backend_mod._get_backend("missing")


def test_column_names_from_collect_schema():
    """Reads names from a collect_schema() method."""
    data = types.SimpleNamespace()
    schema = types.SimpleNamespace(names=lambda: ["a", "b"])
    data.collect_schema = lambda: schema
    assert columns_mod._column_names_from_collect_schema(data) == ["a", "b"]


def test_column_names_from_columns_attr():
    """Reads names from a columns attribute."""
    data = types.SimpleNamespace(columns=("x", "y"))
    assert columns_mod._column_names_from_columns_attr(data) == ["x", "y"]


def test_column_names_from_schema_iterable():
    """Reads names from a schema() iterable of fields."""

    class Field:
        """Simple schema field stub."""

        def __init__(self, name):
            self.name = name

    data = types.SimpleNamespace(schema=lambda: [Field("c"), Field("d")])
    assert columns_mod._column_names_from_schema(data) == ["c", "d"]


def test_validate_required_id_raises():
    """Raises when required id column is missing."""
    data = types.SimpleNamespace(columns=["a", "b"])
    with pytest.raises(ValueError, match="missing required id column"):
        columns_mod._validate_required_id(data, "id")
