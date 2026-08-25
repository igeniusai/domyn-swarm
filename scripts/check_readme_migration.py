# Copyright 2025 iGenius S.p.A
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check that every section cut from the README landed somewhere in docs/.

Run against the pre-cut README (read from git) and the current docs tree. Any
distinctive token -- a command, a flag, a config key, an environment variable --
that the old README mentioned and the site does not is reported as a loss.

Part of the site's content lives only in generated pages: the CLI reference is a
``sphinx-click`` directive, not prose, so its flags exist in the *built* HTML and
in no source file. Point ``--built`` at a built site to count those, which is why
this script is run after ``sphinx-build`` rather than instead of it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys

# Tokens that must survive the migration: commands, flags, config keys and
# environment variables a user might search the site for.
TOKEN_PATTERNS = (
    re.compile(r"--[a-z][a-z0-9-]+"),
    re.compile(r"\bDOMYN_SWARM_[A-Z_]+\b"),
    re.compile(r"\bLEPTON(?:AI)?_[A-Z_]+\b"),
    re.compile(r"\bdomyn-swarm(?: [a-z-]+){1,3}\b"),
)

# Tokens deliberately not carried over, each with the reason. A token belongs
# here only when the site is *correct* without it -- not when documenting it is
# merely inconvenient.
ALLOWED_LOSSES: dict[str, str] = {
    "--help": "sphinx-click omits it by convention; every Click command has it",
    # The token regex cannot tell a command from a sentence that starts with one,
    # nor a real argument from an example name.
    "domyn-swarm only validates their": "prose, not a command",
    "domyn-swarm down my-beautiful-llm-swarm": "an example swarm name",
    "domyn-swarm swarm describe my-swarm-name": "an example swarm name",
}


def readme_at(ref: str) -> str:
    """Return the README as it exists at a git ref."""
    return subprocess.run(
        ["git", "show", f"{ref}:README.md"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def tokens(text: str) -> set[str]:
    """Extract every searchable token from a body of text."""
    found: set[str] = set()
    for pattern in TOKEN_PATTERNS:
        found.update(match.group(0) for match in pattern.finditer(text))
    return found


def docs_corpus(docs_dir: Path, built: Path | None = None) -> str:
    """Concatenate everything a reader could find the tokens in.

    ``_generated`` is included: those tables are the configuration and
    environment reference, generated at build time but real documentation.
    ``superpowers`` is excluded -- it is design notes, not published pages.
    """
    parts = [
        path.read_text(encoding="utf-8")
        for path in sorted(docs_dir.rglob("*.md"))
        if "superpowers" not in path.parts
    ]
    if built is not None and built.is_dir():
        parts.extend(
            path.read_text(encoding="utf-8", errors="replace")
            for path in sorted(built.rglob("*.html"))
        )
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="origin/main", help="git ref holding the pre-cut README")
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    parser.add_argument(
        "--built",
        type=Path,
        default=Path("docs/_build/html"),
        help="a built site, whose generated pages also count as documentation",
    )
    args = parser.parse_args()

    before = tokens(readme_at(args.ref))
    after = (
        docs_corpus(args.docs, args.built) + "\n" + Path("README.md").read_text(encoding="utf-8")
    )

    lost = sorted(t for t in before if t not in after and t not in ALLOWED_LOSSES)
    if lost:
        print("Tokens present in the old README but nowhere in the docs or new README:")
        for token in lost:
            print(f"  {token}")
        print("\nEither document each one, or add it to ALLOWED_LOSSES with a reason.")
        return 1

    print(f"OK: all {len(before)} README tokens accounted for.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
