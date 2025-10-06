from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

# Adjust this import if your module path differs
import domyn_swarm.config.defaults as mod

# --- Helpers -----------------------------------------------------------------


def _write_yaml(p: Path, data) -> None:
    p.write_text(yaml.safe_dump(data, sort_keys=False))


@pytest.fixture(autouse=True)
def _clear_cache_between_tests():
    # Ensure _load_defaults() cache is cleared for each test
    mod.reload_defaults_cache()
    yield
    mod.reload_defaults_cache()


# --- _find_defaults_file ------------------------------------------------------


def test_find_defaults_file_prefers_settings_defaults_file(mocker, tmp_path: Path):
    defaults = tmp_path / "defaults.yaml"
    _write_yaml(defaults, {"a": 1})

    # get_settings().defaults_file points to our file
    mocker.patch.object(
        mod, "get_settings", return_value=SimpleNamespace(defaults_file=str(defaults))
    )
    # Ensure _DEFAULT_FILES does not interfere
    mocker.patch.object(mod, "_DEFAULT_FILES", [])

    p = mod._find_defaults_file()
    assert p == defaults


def test_find_defaults_file_uses_searchers_when_no_settings_value(
    mocker, tmp_path: Path
):
    defaults = tmp_path / "defaults.yaml"
    _write_yaml(defaults, {"a": 1})

    # No explicit defaults_file set in settings
    mocker.patch.object(
        mod, "get_settings", return_value=SimpleNamespace(defaults_file=None)
    )

    # First searcher returns None, second returns our file
    mocker.patch.object(mod, "_DEFAULT_FILES", [lambda: None, lambda: defaults])

    p = mod._find_defaults_file()
    assert p == defaults


def test_find_defaults_file_returns_none_when_nothing_found(mocker):
    mocker.patch.object(
        mod, "get_settings", return_value=SimpleNamespace(defaults_file=None)
    )
    mocker.patch.object(mod, "_DEFAULT_FILES", [lambda: None, lambda: None])
    assert mod._find_defaults_file() is None


# --- _load_defaults (with cache) ---------------------------------------------


def test_load_defaults_reads_yaml_and_caches(mocker, tmp_path: Path):
    defaults = tmp_path / "defaults.yaml"
    _write_yaml(defaults, {"slurm": {"endpoint": {"port": 9000}}})

    mocker.patch.object(
        mod, "get_settings", return_value=SimpleNamespace(defaults_file=str(defaults))
    )
    # fresh cache
    mod.reload_defaults_cache()

    d1 = mod._load_defaults()
    assert d1 == {"slurm": {"endpoint": {"port": 9000}}}

    # Modify file after first load; should still get cached data
    _write_yaml(defaults, {"slurm": {"endpoint": {"port": 9999}}})
    d2 = mod._load_defaults()
    assert d2 == {"slurm": {"endpoint": {"port": 9000}}}  # cached

    # Clear cache and reload
    mod.reload_defaults_cache()
    d3 = mod._load_defaults()
    assert d3 == {"slurm": {"endpoint": {"port": 9999}}}


@pytest.mark.parametrize("payload", [42, ["x", "y"], None])
def test_load_defaults_non_dict_returns_empty(mocker, tmp_path: Path, payload):
    defaults = tmp_path / "defaults.yaml"
    _write_yaml(defaults, payload)

    mocker.patch.object(
        mod, "get_settings", return_value=SimpleNamespace(defaults_file=str(defaults))
    )
    mod.reload_defaults_cache()

    d = mod._load_defaults()
    assert d == {}


# --- _get_by_dots -------------------------------------------------------------


def test_get_by_dots_success():
    d = {"a": {"b": {"c": 123}}}
    assert mod._get_by_dots(d, "a.b.c") == 123


def test_get_by_dots_missing_key_returns_none():
    d = {"a": {"b": {"c": 123}}}
    assert mod._get_by_dots(d, "a.b.x") is None
    assert mod._get_by_dots(d, "x.y") is None


# --- get_default / default_for -----------------------------------------------


def test_get_default_value_from_defaults_file(mocker, tmp_path: Path):
    defaults = tmp_path / "defaults.yaml"
    _write_yaml(defaults, {"slurm": {"endpoint": {"port": 9000}}})

    mocker.patch.object(
        mod, "get_settings", return_value=SimpleNamespace(defaults_file=str(defaults))
    )
    mod.reload_defaults_cache()

    v = mod.get_default("slurm.endpoint.port", fallback=123)
    assert v == 9000


def test_get_default_missing_returns_fallback_when_provided(mocker, tmp_path: Path):
    defaults = tmp_path / "defaults.yaml"
    _write_yaml(defaults, {"slurm": {"endpoint": {"port": 9000}}})

    mocker.patch.object(
        mod, "get_settings", return_value=SimpleNamespace(defaults_file=str(defaults))
    )
    mod.reload_defaults_cache()

    v = mod.get_default("slurm.endpoint.missing", fallback="fallback-value")
    assert v == "fallback-value"


def test_get_default_missing_required_raises(mocker, tmp_path: Path):
    defaults = tmp_path / "defaults.yaml"
    _write_yaml(defaults, {"slurm": {"endpoint": {"port": 9000}}})

    mocker.patch.object(
        mod, "get_settings", return_value=SimpleNamespace(defaults_file=str(defaults))
    )
    mod.reload_defaults_cache()

    with pytest.raises(ValueError):
        mod.get_default("slurm.endpoint.missing", fallback=mod._REQUIRED)


def test_get_default_empty_string_treated_as_missing(mocker, tmp_path: Path):
    defaults = tmp_path / "defaults.yaml"
    _write_yaml(defaults, {"slurm": {"endpoint": {"nginx_image": ""}}})

    mocker.patch.object(
        mod, "get_settings", return_value=SimpleNamespace(defaults_file=str(defaults))
    )
    mod.reload_defaults_cache()

    # Fallback path
    v = mod.get_default("slurm.endpoint.nginx_image", fallback="abc.sif")
    assert v == "abc.sif"

    # Required path
    with pytest.raises(ValueError):
        mod.get_default("slurm.endpoint.nginx_image", fallback=mod._REQUIRED)


def test_default_for_factory_returns_value(mocker, tmp_path: Path):
    defaults = tmp_path / "defaults.yaml"
    _write_yaml(defaults, {"slurm": {"endpoint": {"port": 7777}}})

    mocker.patch.object(
        mod, "get_settings", return_value=SimpleNamespace(defaults_file=str(defaults))
    )
    mod.reload_defaults_cache()

    factory = mod.default_for("slurm.endpoint.port")
    assert factory() == 7777


def test_default_for_factory_returns_fallback_when_missing(mocker, tmp_path: Path):
    defaults = tmp_path / "defaults.yaml"
    _write_yaml(defaults, {"slurm": {"endpoint": {"port": 7777}}})

    mocker.patch.object(
        mod, "get_settings", return_value=SimpleNamespace(defaults_file=str(defaults))
    )
    mod.reload_defaults_cache()

    factory = mod.default_for("slurm.endpoint.missing", fallback="fallback")
    assert factory() == "fallback"


def test_default_for_factory_raises_when_required_and_missing(mocker, tmp_path: Path):
    defaults = tmp_path / "defaults.yaml"
    _write_yaml(defaults, {"slurm": {"endpoint": {"port": 7777}}})

    mocker.patch.object(
        mod, "get_settings", return_value=SimpleNamespace(defaults_file=str(defaults))
    )
    mod.reload_defaults_cache()

    factory = mod.default_for("slurm.endpoint.missing")  # required by default
    with pytest.raises(ValueError):
        factory()
