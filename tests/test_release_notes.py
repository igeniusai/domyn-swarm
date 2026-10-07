# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Tests for the CI helper that renders GitHub release notes from the changelog.

The script lives under ``.github/scripts`` rather than in the package, because it
is only ever run by the release workflow, so it is loaded here by path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / ".github" / "scripts" / "release_notes.py"


def _load():
    spec = importlib.util.spec_from_file_location("release_notes", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


notes = _load()

_CHANGELOG = """\
## v0.3.0 (2026-10-07)

### BREAKING CHANGE

- submit_job requires run=JobRunSpec(...)
and no longer accepts flat parameters. Pass them as in
--job-kwargs.

### Feat

- **slurm**: size job steps

## v0.2.1 (2026-09-25)

### Fix

- **deps**: patch advisories

## v0.2.0 (2026-09-01)

### Feat

- first feature
"""


def test_renders_the_section_without_its_heading_and_links_the_previous_tag():
    rendered = notes.release_notes(_CHANGELOG, "v0.3.0", repo="org/repo")

    assert rendered == (
        "### BREAKING CHANGE\n"
        "\n"
        "- submit_job requires run=JobRunSpec(...) and no longer accepts flat "
        "parameters. Pass them as in --job-kwargs.\n"
        "\n"
        "### Feat\n"
        "\n"
        "- **slurm**: size job steps\n"
        "\n"
        "**Full Changelog**: https://github.com/org/repo/compare/v0.2.1...v0.3.0\n"
    )


def test_a_middle_section_stops_at_the_next_heading():
    rendered = notes.release_notes(_CHANGELOG, "v0.2.1", repo="org/repo")

    assert rendered.startswith("### Fix\n\n- **deps**: patch advisories\n\n")
    assert "first feature" not in rendered
    assert rendered.endswith("compare/v0.2.0...v0.2.1\n")


def test_the_oldest_section_has_no_compare_link():
    rendered = notes.release_notes(_CHANGELOG, "v0.2.0", repo="org/repo")

    assert rendered == "### Feat\n\n- first feature\n"


def test_an_unknown_tag_fails():
    with pytest.raises(SystemExit, match=r"v9\.9\.9"):
        notes.release_notes(_CHANGELOG, "v9.9.9", repo="org/repo")


def test_every_release_in_the_repository_changelog_renders():
    changelog = (_ROOT / "CHANGELOG.md").read_text()
    tags = [line.split()[1] for line in changelog.splitlines() if line.startswith("## v")]

    for tag in tags:
        rendered = notes.release_notes(changelog, tag, repo="org/repo")
        assert rendered.strip(), tag
        assert "\n## " not in rendered, tag


def test_an_empty_section_renders_only_the_compare_link():
    changelog = "## v0.2.0 (2026-10-07)\n\n## v0.1.0 (2026-10-01)\n\n### Feat\n\n- start\n"

    rendered = notes.release_notes(changelog, "v0.2.0", repo="org/repo")

    assert rendered == "**Full Changelog**: https://github.com/org/repo/compare/v0.1.0...v0.2.0\n"
