#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Render the GitHub release notes for a tag from its ``CHANGELOG.md`` section.

Commitizen writes the changelog on every bump, so it is the single source for
release notes. The section heading is dropped because the release title already
shows the version, and the notes end with a compare link to the previous
release, which is the next section down.

Commitizen copies multi-line commit bodies, such as a ``BREAKING CHANGE``
footer, into a list item with their hard line breaks. GitHub renders each of
those breaks, so continuation lines are joined back into their list item.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_SECTION_PREFIX = "## "


def _version_of(heading: str) -> str:
    return heading.removeprefix(_SECTION_PREFIX).split()[0]


def _join_continuations(lines: list[str]) -> list[str]:
    joined: list[str] = []
    for line in lines:
        starts_block = not line or line.startswith(("#", "- "))
        if joined and joined[-1].startswith("- ") and not starts_block:
            joined[-1] = f"{joined[-1]} {line.strip()}"
        else:
            joined.append(line)
    return joined


def release_notes(changelog: str, tag: str, *, repo: str) -> str:
    """Return the release notes for ``tag``.

    Raises:
        SystemExit: If the changelog has no section for ``tag``.
    """
    lines = changelog.splitlines()
    headings = [i for i, line in enumerate(lines) if line.startswith(_SECTION_PREFIX)]
    position = next((n for n, i in enumerate(headings) if _version_of(lines[i]) == tag), None)
    if position is None:
        raise SystemExit(f"CHANGELOG.md has no section for {tag}")

    start = headings[position] + 1
    end = headings[position + 1] if position + 1 < len(headings) else len(lines)
    parts = ["\n".join(_join_continuations(lines[start:end])).strip()]

    if position + 1 < len(headings):
        previous = _version_of(lines[headings[position + 1]])
        parts.append(f"**Full Changelog**: https://github.com/{repo}/compare/{previous}...{tag}")
    return "\n\n".join(part for part in parts if part) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render the GitHub release notes for a tag from CHANGELOG.md."
    )
    parser.add_argument("tag", help="Release tag, e.g. v0.34.0.")
    parser.add_argument("--repo", required=True, help="GitHub repository as owner/name.")
    parser.add_argument("--changelog", type=Path, default=Path("CHANGELOG.md"))
    args = parser.parse_args(argv)

    sys.stdout.write(release_notes(args.changelog.read_text(), args.tag, repo=args.repo))
    return 0


if __name__ == "__main__":
    sys.exit(main())
