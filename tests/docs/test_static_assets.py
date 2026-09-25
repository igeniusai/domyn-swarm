# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Stylesheets under ``docs/_static`` must reference only files the site ships.

A missing font or icon does not fail the Sphinx build: the browser falls back to
a system font or paints an empty box. The fonts are self-hosted so that no page
makes a request to a third-party origin.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

STATIC = Path(__file__).resolve().parents[2] / "docs" / "_static"
_CSS_URL = re.compile(r"""url\(\s*["']?([^"')]+?)["']?\s*\)""")


def _stylesheets() -> list[Path]:
    return sorted(STATIC.rglob("*.css"))


@pytest.mark.parametrize("css", _stylesheets(), ids=lambda p: str(p.relative_to(STATIC)))
def test_every_css_url_is_a_shipped_file(css: Path) -> None:
    urls = [
        u for u in _CSS_URL.findall(css.read_text(encoding="utf-8")) if not u.startswith("data:")
    ]
    remote = [u for u in urls if "://" in u or u.startswith("//")]
    missing = [u for u in urls if u not in remote and not (css.parent / u).is_file()]
    assert not remote, f"{css.name} loads from another origin: {remote}"
    assert not missing, f"{css.name} references missing files: {missing}"


def test_brand_stylesheet_self_hosts_dm_sans() -> None:
    brand = (STATIC / "brand.css").read_text(encoding="utf-8")
    assert '"DM Sans"' in brand
    assert "fonts/dm-sans-latin.woff2" in brand


def test_self_hosted_fonts_ship_their_license() -> None:
    assert (STATIC / "fonts" / "OFL.txt").is_file()
