#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Turn a Cobertura ``coverage.xml`` into a shields.io badge payload.

The file this writes is published to the repository's ``badges`` branch and read
by shields.io over raw.githubusercontent.com, which is why it has to be plain
JSON at a stable path rather than anything served dynamically.

The payload deliberately satisfies two shields formats at once:

* the **endpoint** badge reads ``schemaVersion``/``label``/``message``/``color``,
  so the colour is decided here and tracks the number;
* the **dynamic JSON** badge reads the numeric ``coverage`` field via a query.

Switching between the two is therefore a README edit, with no CI change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

# Descending floors, mirroring the colour scale shields uses for its own
# coverage badges.
_SCALE: tuple[tuple[float, str], ...] = (
    (95, "brightgreen"),
    (90, "green"),
    (75, "yellowgreen"),
    (60, "yellow"),
    (40, "orange"),
    (0, "red"),
)


def colour_for(percent: float) -> str:
    """Return the shields colour name for a coverage percentage."""
    for floor, colour in _SCALE:
        if percent >= floor:
            return colour
    return "red"


def percent_from(xml_path: Path) -> float:
    """Read the overall line coverage percentage out of a Cobertura report."""
    root = ET.parse(xml_path).getroot()
    rate = root.get("line-rate")
    if rate is None:
        raise SystemExit(f"{xml_path}: no line-rate attribute; is this a Cobertura report?")
    return round(float(rate) * 100, 1)


def payload(percent: float) -> dict[str, object]:
    """Build the badge document for both shields formats."""
    shown = f"{percent:.0f}%" if float(percent).is_integer() else f"{percent}%"
    return {
        "schemaVersion": 1,
        "label": "coverage",
        "message": shown,
        "color": colour_for(percent),
        "coverage": percent,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml", type=Path, help="path to coverage.xml")
    parser.add_argument("-o", "--output", type=Path, help="write JSON here instead of stdout")
    args = parser.parse_args(argv)

    doc = payload(percent_from(args.xml))
    text = json.dumps(doc, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
