# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Template overrides must track the pydata-sphinx-theme originals.

``docs/_templates/theme-switcher.html`` is a copy of the theme's component with
the Font Awesome icons swapped for the brand icons. When a theme upgrade changes
the original anywhere else, the sync test fails so the copy can be rebuilt.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

pydata_sphinx_theme = pytest.importorskip(
    "pydata_sphinx_theme", reason="requires the docs dependency group"
)

DOCS = Path(__file__).resolve().parents[2] / "docs"
OVERRIDE = DOCS / "_templates" / "theme-switcher.html"
UPSTREAM = (
    Path(pydata_sphinx_theme.__file__).parent
    / "theme"
    / "pydata_sphinx_theme"
    / "components"
    / "theme-switcher.html"
)
_ICON = re.compile(r"<(i|span) ([^>]*)></\1>")
_ATTRIBUTE = re.compile(r'([\w-]+)="([^"]*)"')
_GLYPH_CLASS = re.compile(r"fa|fa-[\w-]+|domyn-icon(--[\w-]+)?")
_MODIFIER = re.compile(r"\bdomyn-icon--([a-z_-]+)")


def _without_icon_glyphs(template: str) -> str:
    """Reduce each icon element to the classes and attributes the theme relies on.

    Only the glyph classes and the added ``aria-hidden`` differ on purpose, so
    every other class and attribute on an icon still has to match upstream.
    """

    def reduce(icon: re.Match[str]) -> str:
        attributes = dict(_ATTRIBUTE.findall(icon.group(2)))
        attributes.pop("aria-hidden", None)
        classes = attributes.pop("class", "").split()
        kept_classes = " ".join(c for c in classes if not _GLYPH_CLASS.fullmatch(c))
        kept = "".join(f' {name}="{value}"' for name, value in sorted(attributes.items()))
        return f'<icon class="{kept_classes}"{kept}>'

    return _ICON.sub(reduce, template)


def test_theme_switcher_override_changes_only_the_icons() -> None:
    upstream = _without_icon_glyphs(UPSTREAM.read_text(encoding="utf-8"))
    override = _without_icon_glyphs(OVERRIDE.read_text(encoding="utf-8"))
    assert override == upstream


def test_sync_check_notices_a_renamed_icon_hook() -> None:
    """The theme's CSS shows one icon per mode through ``.theme-switch[data-mode]``."""
    renamed = UPSTREAM.read_text(encoding="utf-8").replace(
        'class="theme-switch ', 'class="pst-mode-icon '
    )
    override = OVERRIDE.read_text(encoding="utf-8")
    assert _without_icon_glyphs(renamed) != _without_icon_glyphs(override)


@pytest.mark.parametrize(
    ("mode", "icon"), [("light", "sun"), ("dark", "moon"), ("auto", "contrast")]
)
def test_each_mode_shows_its_brand_icon_in_button_and_menu(mode: str, icon: str) -> None:
    text = OVERRIDE.read_text(encoding="utf-8")
    assert f'domyn-icon--{icon}" data-mode="{mode}"' in text
    assert f'data-mode="{mode}"><span class="domyn-icon domyn-icon--{icon}' in text


def test_every_icon_modifier_in_templates_has_a_css_rule() -> None:
    brand = (DOCS / "_static" / "brand.css").read_text(encoding="utf-8")
    used = {
        modifier
        for path in (DOCS / "_templates").glob("*.html")
        for modifier in _MODIFIER.findall(path.read_text(encoding="utf-8"))
    }
    undefined = sorted(m for m in used if f".domyn-icon--{m} " not in brand)
    assert used, "expected the templates to use brand icons"
    assert not undefined, f"brand.css has no rule for: {undefined}"
