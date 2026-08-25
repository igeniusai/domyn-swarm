# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Every source file must carry the current SPDX licence header.

The one-off rewriter that converted the legacy Apache boilerplate has been
deleted -- it had one job and did it. This guard is what remains: it stops a new
file landing with no header, and stops the old entity name coming back.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

REPO = Path(__file__).resolve().parents[2]
EXPECTED = (
    "# SPDX-FileCopyrightText: 2025-2026 Domyn",
    "# SPDX-License-Identifier: Apache-2.0",
)
ROOTS = ("src", "tests", "examples", "docs")
SKIP_DIRS = {".venv", "_build", "__pycache__", ".git", "migrations"}

# This file must name the old entity in order to detect it.
_MAY_NAME_THE_OLD_ENTITY = {Path("tests/docs/test_license_headers.py")}


def _sources() -> list[Path]:
    """Every first-party Python file that should carry a header.

    ``migrations`` is skipped: Alembic generates those from ``script.py.mako``,
    so the template is the thing to keep current.
    """
    found: list[Path] = []
    for root in ROOTS:
        for path in sorted((REPO / root).rglob("*.py")):
            if SKIP_DIRS & set(path.parts) or path.stat().st_size == 0:
                continue
            found.append(path)
    return found


def _header_lines(text: str) -> tuple[str, ...]:
    """The two header lines, looking past a shebang or encoding declaration.

    Both are positional -- a shebang only works on line 1 -- so the header sits
    below them rather than at the very top of every file.
    """
    lines = text.split("\n")
    start = 0
    if lines and lines[0].startswith("#!"):
        start = 1
    if len(lines) > start and re.search(r"coding[:=]", lines[start]):
        start += 1
    return tuple(lines[start : start + 2])


@pytest.mark.parametrize("path", _sources(), ids=lambda p: str(p.relative_to(REPO)))
def test_file_has_the_spdx_header(path: Path) -> None:
    head = _header_lines(path.read_text(encoding="utf-8"))
    assert head == EXPECTED, f"{path} has a stale or missing licence header"


def test_a_shebang_stays_on_the_first_line() -> None:
    """A header above a shebang would stop the script executing."""
    scripts = [p for p in _sources() if p.read_text(encoding="utf-8").startswith("#!")]
    assert scripts, "expected at least one executable script to guard"
    for path in scripts:
        assert _header_lines(path.read_text(encoding="utf-8")) == EXPECTED


def test_no_file_still_names_the_old_entity() -> None:
    stale = sorted(
        str(p.relative_to(REPO))
        for p in _sources()
        if p.relative_to(REPO) not in _MAY_NAME_THE_OLD_ENTITY
        and "iGenius" in p.read_text(encoding="utf-8")
    )
    assert not stale, f"files still naming the old entity: {stale}"


def test_the_migration_template_is_current() -> None:
    """Alembic copies this template verbatim, so a stale header propagates."""
    template = REPO / "src" / "domyn_swarm" / "core" / "state" / "migrations" / "script.py.mako"
    assert template.read_text(encoding="utf-8").startswith(EXPECTED[0])
