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

"""Sphinx configuration for the domyn-swarm documentation site."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path
import sys

DOCS_DIR = Path(__file__).parent
sys.path.insert(0, str(DOCS_DIR / "_ext"))

project = "domyn-swarm"
author = "iGenius S.p.A"
copyright = "2025, iGenius S.p.A"


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

autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "undoc-members": False,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
}

html_theme = "pydata_sphinx_theme"
html_title = "domyn-swarm"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_baseurl = "https://igeniusai.github.io/domyn-swarm/"

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

html_context = {
    "github_user": "igeniusai",
    "github_repo": "domyn-swarm",
    "github_version": "main",
    "doc_path": "docs",
}
