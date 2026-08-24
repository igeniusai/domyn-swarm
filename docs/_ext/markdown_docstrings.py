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

"""Let autodoc accept the Markdown-flavoured docstrings this codebase writes.

Docstrings here use Markdown conventions — fenced code blocks and single-backtick
inline code — inside otherwise Google-style sections. Napoleon converts the
section structure, but the Markdown inline syntax survives into reStructuredText,
where a fence is invalid and a single backtick opens an interpreted-text role
that never closes. Building with ``-W`` then fails on docstrings that are
perfectly readable in an editor.

Rewriting the docstrings in the source would fight the project's convention and
regress the moment someone writes another one, so the conversion happens at build
time instead, on the ``autodoc-process-docstring`` event.

Bold is left alone: ``**text**`` means the same thing in both syntaxes.
"""

from __future__ import annotations

import re
from typing import Any

from sphinx.application import Sphinx

_FENCE = re.compile(r"^(?P<indent>[ \t]*)```(?P<lang>[\w+-]*)[ \t]*$")
_INLINE_CODE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")


def docstring_to_rst(lines: list[str]) -> list[str]:
    """Convert Markdown-isms in a docstring to reStructuredText.

    Args:
        lines: The docstring's lines, as autodoc supplies them.

    Returns:
        The converted lines. Content inside a fenced block is copied verbatim
        apart from indentation, so backticks in example code are preserved.
    """
    out: list[str] = []
    index = 0

    while index < len(lines):
        fence = _FENCE.match(lines[index])
        if fence is None:
            out.append(_INLINE_CODE.sub(r"``\1``", lines[index]))
            index += 1
            continue

        indent = fence.group("indent")
        language = fence.group("lang") or "text"

        body: list[str] = []
        index += 1
        while index < len(lines) and _FENCE.match(lines[index]) is None:
            body.append(lines[index])
            index += 1
        index += 1  # skip the closing fence, or run off the end

        # RST requires a blank line before a directive; Markdown does not, so a
        # fence can sit directly under a "**Heading:**" line.
        if out and out[-1].strip():
            out.append("")
        out.append(f"{indent}.. code-block:: {language}")
        out.append("")
        # Three extra spaces put the body inside the directive while preserving
        # the code's own relative indentation.
        out.extend(f"   {line}" if line.strip() else "" for line in body)
        out.append("")

    return out


def _process_docstring(
    app: Sphinx,
    what: str,
    name: str,
    obj: Any,
    options: Any,
    lines: list[str],
) -> None:
    """Rewrite ``lines`` in place; autodoc reads the list back out.

    Every parameter before ``lines`` is required by the ``autodoc-process-docstring``
    signature and is deliberately unused.
    """
    lines[:] = docstring_to_rst(lines)


def setup(app: Sphinx) -> dict[str, object]:
    # Priority 0 runs before Napoleon's own handler, so Napoleon sees valid RST
    # when it builds its field lists.
    app.connect("autodoc-process-docstring", _process_docstring, priority=0)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
