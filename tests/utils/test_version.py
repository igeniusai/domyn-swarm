from importlib.metadata import PackageNotFoundError

from domyn_swarm.utils.version import get_version


def test_get_version_from_installed(monkeypatch):
    """Returns the package version when importlib.metadata resolves it.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """

    def _fake_version(_: str) -> str:
        """Return a fixed version string."""
        return "1.2.3"

    monkeypatch.setattr("domyn_swarm.utils.version.version", _fake_version)
    assert get_version() == "1.2.3"


def test_get_version_from_pyproject(monkeypatch, tmp_path):
    """Falls back to pyproject.toml when the package is not installed.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary directory fixture.
    """

    def _raise(_: str) -> str:
        """Raise as if the package is not installed."""
        raise PackageNotFoundError("missing")

    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nversion = "9.9.9"\n')
    monkeypatch.setattr("domyn_swarm.utils.version.version", _raise)
    monkeypatch.chdir(tmp_path)
    assert get_version() == "9.9.9"


def test_get_version_missing_key_returns_unknown(monkeypatch, tmp_path):
    """Returns the default when pyproject lacks a version entry.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary directory fixture.
    """

    def _raise(_: str) -> str:
        """Raise as if the package is not installed."""
        raise PackageNotFoundError("missing")

    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'domyn-swarm'\n")
    monkeypatch.setattr("domyn_swarm.utils.version.version", _raise)
    monkeypatch.chdir(tmp_path)
    assert get_version() == "0.0.0+unknown"
