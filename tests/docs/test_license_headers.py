# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Every source file must carry the current SPDX licence header."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
EXPECTED = (
    "# SPDX-FileCopyrightText: 2025-2026 Domyn",
    "# SPDX-License-Identifier: Apache-2.0",
)
ROOTS = ("src", "tests", "scripts", "examples", "docs/_ext")
SKIP_DIRS = {".venv", "_build", "__pycache__", ".git", "migrations"}

_SCRIPT = REPO / "scripts" / "update_license_headers.py"
_spec = importlib.util.spec_from_file_location("update_license_headers", _SCRIPT)
assert _spec is not None and _spec.loader is not None
update_license_headers = importlib.util.module_from_spec(_spec)
sys.modules["update_license_headers"] = update_license_headers
_spec.loader.exec_module(update_license_headers)
rewrite = update_license_headers.rewrite
HEADER = update_license_headers.HEADER

_LEGACY_SAMPLE = (
    "# Copyright 2025 iGenius S.p.A\n"
    "#\n"
    '# Licensed under the Apache License, Version 2.0 (the "License");\n'
    "# you may not use this file except in compliance with the License.\n"
    "# limitations under the License.\n"
    "\n"
    '"""Docstring."""\n'
)


def _sources() -> list[Path]:
    """Every first-party Python file that should carry a header.

    ``migrations`` is skipped: Alembic generates those from
    ``script.py.mako``, so the template is the thing to keep current.
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
    """Inserting a header above a shebang would stop the script executing."""
    scripts = [p for p in _sources() if p.read_text(encoding="utf-8").startswith("#!")]
    assert scripts, "expected at least one executable script to guard"
    for path in scripts:
        assert _header_lines(path.read_text(encoding="utf-8")) == EXPECTED


# The rewriter and this test must both name the old entity in order to detect it.
_MAY_NAME_THE_OLD_ENTITY = {
    Path("scripts/update_license_headers.py"),
    Path("tests/docs/test_license_headers.py"),
}


def test_no_file_still_names_the_old_entity() -> None:
    stale = sorted(
        str(p.relative_to(REPO))
        for p in _sources()
        if p.relative_to(REPO) not in _MAY_NAME_THE_OLD_ENTITY
        and "iGenius" in p.read_text(encoding="utf-8")
    )
    assert not stale, f"files still naming the old entity: {stale}"


def test_rewriter_inserts_a_header_below_a_shebang() -> None:
    text = '#!/usr/bin/env python3\n"""Docstring."""\n'
    out = rewrite(text)
    assert out.startswith("#!/usr/bin/env python3\n")
    assert _header_lines(out) == EXPECTED


def test_rewriter_inserts_a_header_when_the_file_has_none() -> None:
    out = rewrite("import typer\n")
    assert _header_lines(out) == EXPECTED
    assert out.endswith("import typer\n")


def test_rewriter_leaves_an_empty_file_alone() -> None:
    assert rewrite("") == ""
    assert rewrite("\n") == "\n"


def test_the_migration_template_is_current() -> None:
    """Alembic copies this template verbatim, so a stale header propagates."""
    template = REPO / "src" / "domyn_swarm" / "core" / "state" / "migrations" / "script.py.mako"
    assert template.read_text(encoding="utf-8").startswith(EXPECTED[0])


def test_rewriter_replaces_the_legacy_block() -> None:
    out = rewrite(_LEGACY_SAMPLE)
    assert out.startswith("# SPDX-FileCopyrightText: 2025-2026 Domyn\n")
    assert "iGenius" not in out
    assert out.endswith('"""Docstring."""\n')


def test_rewriter_keeps_exactly_one_blank_line_after_the_header() -> None:
    out = rewrite(_LEGACY_SAMPLE)
    assert out.split("\n")[:4] == [
        "# SPDX-FileCopyrightText: 2025-2026 Domyn",
        "# SPDX-License-Identifier: Apache-2.0",
        "",
        '"""Docstring."""',
    ]


def test_rewriter_is_idempotent() -> None:
    once = rewrite(_LEGACY_SAMPLE)
    assert rewrite(once) == once


def test_rewriter_only_touches_a_header_at_the_top() -> None:
    """A quoted copy of the boilerplate mid-file is data, not a header."""
    text = 'HELP = """\n' + _LEGACY_SAMPLE + '"""\n'
    out = rewrite(text)
    assert out == HEADER + "\n" + text, "the quoted boilerplate must survive verbatim"
