# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Regenerate ``switcher.json`` and the root redirect for the assembled site.

``switcher.json`` must live at a URL that is stable across versions, because
every published version fetches the same file to build its version dropdown. It
is therefore owned by the deploy step rather than by any one version's Sphinx
build. Run this against the assembled site root, once every version directory
has been built into it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

BASE_URL = "https://domynswarm.domyn.com"
VERSION_DIR = re.compile(r"^v(\d+)\.(\d+)$")


def _sort_key(path: Path) -> tuple[int, int]:
    match = VERSION_DIR.match(path.name)
    assert match is not None
    major, minor = match.groups()
    return int(major), int(minor)


def build_entries(root: Path) -> list[dict[str, object]]:
    """Build the switcher entries for an assembled site, newest release first.

    Args:
        root: The assembled site root to scan for published versions.

    Returns:
        One entry per published version. ``latest`` leads if it exists, then
        releases newest first, with the newest marked ``preferred`` so the theme
        knows which one is current.
    """
    entries: list[dict[str, object]] = []

    if (root / "latest").is_dir():
        entries.append(
            {
                "name": "latest (main)",
                "version": "latest",
                "url": f"{BASE_URL}/latest/",
            }
        )

    releases = sorted(
        (p for p in root.iterdir() if p.is_dir() and VERSION_DIR.match(p.name)),
        key=_sort_key,
        reverse=True,
    )
    for index, path in enumerate(releases):
        entry: dict[str, object] = {
            "name": path.name.removeprefix("v"),
            "version": path.name,
            "url": f"{BASE_URL}/{path.name}/",
        }
        if index == 0:
            entry["name"] = f"{entry['name']} (stable)"
            entry["preferred"] = True
        entries.append(entry)

    return entries


def write_root_redirect(root: Path, entries: list[dict[str, object]]) -> None:
    """Point the site root at the preferred version, falling back to ``latest``.

    There is deliberately no ``stable/`` directory: publishing one build at two
    URLs would serve duplicate content, so the root redirects into the newest
    version directory instead.
    """
    target = next(
        (e for e in entries if e.get("preferred")),
        next(iter(entries), None),
    )
    if target is None:
        return

    url = str(target["url"])
    (root / "index.html").write_text(
        "<!doctype html>\n"
        '<meta charset="utf-8">\n'
        f'<meta http-equiv="refresh" content="0; url={url}">\n'
        f'<link rel="canonical" href="{url}">\n'
        "<title>domyn-swarm documentation</title>\n"
        f'<p>Redirecting to <a href="{url}">the current documentation</a>.</p>\n',
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="path to the assembled site root")
    args = parser.parse_args()

    entries = build_entries(args.root)
    (args.root / "switcher.json").write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    write_root_redirect(args.root, entries)

    print(f"switcher.json written with {len(entries)} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
