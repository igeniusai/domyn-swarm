# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Bring every source file's licence header to the SPDX short form.

The licence does not change -- it remains Apache-2.0, and ``LICENSE`` keeps the
full text. Only the per-file marker and the attributed entity change, to the
short form that REUSE, scancode and GitHub's licence detection read.

Two jobs, because the repository turned out to hold both: replacing the legacy
Apache boilerplate where it exists, and inserting a header into files that never
carried one. Either way a shebang stays on line 1, where the kernel needs it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

HEADER = "# SPDX-FileCopyrightText: 2025-2026 Domyn\n# SPDX-License-Identifier: Apache-2.0\n"

# A shebang and an encoding declaration are positional: both must stay above the
# header rather than being displaced by it.
PREFIX = re.compile(r"\A(?:#![^\n]*\n)?(?:#[^\n]*coding[:=][^\n]*\n)?")

# Anchored after that prefix: a quoted copy of the boilerplate further down the
# file is data, not a header, and must be left alone.
OLD_BLOCK = re.compile(
    r"\A#\s*Copyright \d{4} iGenius S\.p\.A\n(?:#.*\n)*?#\s*limitations under the License\.\n"
)


def rewrite(text: str) -> str:
    """Return ``text`` carrying the current SPDX header.

    Replaces the legacy block if present, inserts the header if the file has no
    header at all, and returns the text unchanged if it is already current.
    """
    prefix = PREFIX.match(text).group(0)  # type: ignore[union-attr]
    body = text[len(prefix) :]

    if body.startswith("# SPDX-FileCopyrightText"):
        return text

    replaced, count = OLD_BLOCK.subn(HEADER, body, count=1)
    if count:
        # The old block was followed by a blank line; keep exactly one.
        body = re.sub(r"(# SPDX-License-Identifier: Apache-2\.0\n)\n+", r"\1\n", replaced, count=1)
    elif body.strip():
        body = f"{HEADER}\n{body}"
    else:
        # An empty or whitespace-only file carries no code to licence.
        return text

    return prefix + body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report files that would change and exit non-zero, without writing",
    )
    args = parser.parse_args()

    stale: list[Path] = []
    for path in args.paths:
        original = path.read_text(encoding="utf-8")
        updated = rewrite(original)
        if updated == original:
            continue
        stale.append(path)
        if not args.check:
            path.write_text(updated, encoding="utf-8")

    if args.check:
        for path in stale:
            print(f"stale header: {path}")
        print(f"{len(stale)} files would change")
        return 1 if stale else 0

    print(f"rewrote {len(stale)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
