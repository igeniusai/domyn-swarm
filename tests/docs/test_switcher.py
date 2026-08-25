# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Tests for the gh-pages version switcher generator."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "update_switcher.py"
_spec = importlib.util.spec_from_file_location("update_switcher", _SCRIPT)
assert _spec is not None and _spec.loader is not None
update_switcher = importlib.util.module_from_spec(_spec)
sys.modules["update_switcher"] = update_switcher
_spec.loader.exec_module(update_switcher)


@pytest.fixture
def site(tmp_path: Path) -> Path:
    for name in ("latest", "v0.28", "v0.29", "v0.9", "_static", "not-a-version"):
        (tmp_path / name).mkdir()
    return tmp_path


def test_latest_comes_first(site: Path) -> None:
    assert update_switcher.build_entries(site)[0]["version"] == "latest"


def test_versions_are_newest_first(site: Path) -> None:
    entries = update_switcher.build_entries(site)
    versions = [e["version"] for e in entries if e["version"] != "latest"]
    assert versions == ["v0.29", "v0.28", "v0.9"]


def test_newest_release_is_preferred(site: Path) -> None:
    preferred = [e for e in update_switcher.build_entries(site) if e.get("preferred")]
    assert len(preferred) == 1
    assert preferred[0]["version"] == "v0.29"


def test_non_version_directories_are_ignored(site: Path) -> None:
    names = {e["version"] for e in update_switcher.build_entries(site)}
    assert "not-a-version" not in names
    assert "_static" not in names


def test_urls_are_absolute_and_trailing_slashed(site: Path) -> None:
    for entry in update_switcher.build_entries(site):
        url = str(entry["url"])
        assert url.startswith("https://igeniusai.github.io/domyn-swarm/")
        assert url.endswith("/")


def test_root_redirect_points_at_the_preferred_version(site: Path) -> None:
    entries = update_switcher.build_entries(site)
    update_switcher.write_root_redirect(site, entries)
    html = (site / "index.html").read_text()
    assert "https://igeniusai.github.io/domyn-swarm/v0.29/" in html
    assert "http-equiv" in html


def test_root_redirect_falls_back_to_latest_when_no_release(tmp_path: Path) -> None:
    (tmp_path / "latest").mkdir()
    entries = update_switcher.build_entries(tmp_path)
    update_switcher.write_root_redirect(tmp_path, entries)
    assert "/latest/" in (tmp_path / "index.html").read_text()


def test_empty_site_does_not_crash(tmp_path: Path) -> None:
    entries = update_switcher.build_entries(tmp_path)
    assert entries == []
    update_switcher.write_root_redirect(tmp_path, entries)
    assert not (tmp_path / "index.html").exists()
