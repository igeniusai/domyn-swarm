# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Tests for the CI helper that renders the coverage badge payload.

The script lives under ``.github/scripts`` rather than in the package, because it
is only ever run by the workflow, so it is loaded here by path.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "coverage_badge.py"


def _load():
    spec = importlib.util.spec_from_file_location("coverage_badge", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


badge = _load()


@pytest.mark.parametrize(
    ("percent", "expected"),
    [
        (0, "red"),
        (39.9, "red"),
        (40, "orange"),
        (59.9, "orange"),
        (60, "yellow"),
        (74.9, "yellow"),
        (75, "yellowgreen"),
        (89.9, "yellowgreen"),
        (90, "green"),
        (94.9, "green"),
        (95, "brightgreen"),
        (100, "brightgreen"),
    ],
)
def test_colour_boundaries(percent, expected):
    assert badge.colour_for(percent) == expected


def test_percent_read_from_cobertura(tmp_path):
    xml = tmp_path / "coverage.xml"
    xml.write_text('<?xml version="1.0" ?><coverage line-rate="0.8012"></coverage>')
    assert badge.percent_from(xml) == 80.1


def test_missing_line_rate_is_an_error(tmp_path):
    xml = tmp_path / "coverage.xml"
    xml.write_text('<?xml version="1.0" ?><coverage></coverage>')
    with pytest.raises(SystemExit):
        badge.percent_from(xml)


def test_payload_serves_both_shields_formats():
    doc = badge.payload(80.1)
    # endpoint badge
    assert doc["schemaVersion"] == 1
    assert doc["label"] == "coverage"
    assert doc["message"] == "80.1%"
    assert doc["color"] == "yellowgreen"
    # dynamic JSON badge reads the bare number
    assert doc["coverage"] == 80.1


def test_whole_percentages_drop_the_decimal():
    assert badge.payload(80.0)["message"] == "80%"


def test_cli_writes_a_file(tmp_path):
    xml = tmp_path / "coverage.xml"
    xml.write_text('<?xml version="1.0" ?><coverage line-rate="0.5"></coverage>')
    out = tmp_path / "badge.json"
    assert badge.main([str(xml), "-o", str(out)]) == 0
    assert json.loads(out.read_text())["message"] == "50%"
