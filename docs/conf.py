# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Sphinx configuration for the domyn-swarm documentation site."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version
import os
from pathlib import Path
import sys

DOCS_DIR = Path(__file__).parent
sys.path.insert(0, str(DOCS_DIR / "_ext"))

# Build the Click group now, before autodoc runs. autodoc's own imports leave the
# interpreter in a state where importing domyn_swarm.cli.job_helpers raises a
# Pydantic schema-generation error, which would make sphinx-click fail to import
# the group and silently empty the CLI reference. Importing here caches the built
# group in sys.modules, so sphinx-click picks it up already constructed.
import cli_app  # noqa: E402, F401

project = "domyn-swarm"
author = "Domyn"
copyright = "2025-2026, Domyn"


def _release() -> str:
    try:
        return _pkg_version("domyn-swarm")
    except PackageNotFoundError:
        return "0.0.0"


release = _release()
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_click",
    "gen_config_reference",
]

exclude_patterns = [
    "_build",
    "_generated/**",
    "superpowers/**",
    "Thumbs.db",
    ".DS_Store",
]

myst_enable_extensions = [
    "attrs_inline",
    "colon_fence",
    "deflist",
    "fieldlist",
    "substitution",
]
myst_heading_anchors = 3

# Docstrings write inline code with single backticks, as Markdown does. RST would
# otherwise read those as interpreted text with the default role; making that role
# "literal" renders them as code, which is what was meant. This covers 72 docstrings
# across 33 files without any conversion step.
default_role = "literal"

autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "undoc-members": False,
}
# Google-style is the project's docstring convention; NumPy parsing stays off so
# a NumPy-style docstring fails the build rather than rendering badly.
napoleon_google_docstring = True
napoleon_numpy_docstring = False
# Render Attributes: sections as :ivar: fields rather than separate object
# descriptions, which would duplicate the real class attributes autodoc finds.
napoleon_use_ivar = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
}

html_theme = "pydata_sphinx_theme"
html_title = "domyn-swarm"
# The logo lives at the repository root because the README renders it on GitHub
# too. Sphinx merges every entry here into a single _static/, so listing both
# keeps one copy of the asset rather than a copy per consumer.
html_static_path = ["_static", "../static"]
html_css_files = ["custom.css"]
# The Pages custom domain, not the github.io URL: github.io 301-redirects to
# it, so canonical links and the switcher must name the destination.
html_baseurl = "https://domynswarm.domym.com/"

html_theme_options = {
    "logo": {
        "image_light": "_static/domyn-swarm-logo-primary.svg",
        "image_dark": "_static/domyn-swarm-logo-white.svg",
        "alt_text": "domyn-swarm documentation",
    },
    "github_url": "https://github.com/igeniusai/domyn-swarm",
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "show_prev_next": True,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
}

# Which version this build represents. The deploy workflow sets it to "latest" for
# main and "vX.Y" for a release tag; it must match the "version" field the
# switcher emits for the dropdown to highlight the right entry.
DOCS_VERSION = os.environ.get("DOCS_VERSION", "latest")

# switcher.json lives at the gh-pages root, not inside a version directory, because
# every published version fetches the same file. It is written by
# docs/update_switcher.py during deployment.
html_theme_options["switcher"] = {
    "json_url": f"{html_baseurl}switcher.json",
    "version_match": DOCS_VERSION,
}
html_theme_options["navbar_end"] = [
    "version-switcher",
    "theme-switcher",
    "navbar-icon-links",
]
# Tell a reader who arrived on an old version from a search engine that it is old.
html_theme_options["show_version_warning_banner"] = True

html_context = {
    "github_user": "igeniusai",
    "github_repo": "domyn-swarm",
    "github_version": "main",
    "doc_path": "docs",
}
