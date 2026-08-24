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

"""Tests for the Markdown-to-reStructuredText docstring converter."""

from __future__ import annotations

from markdown_docstrings import docstring_to_rst


def _convert(text: str) -> list[str]:
    return docstring_to_rst(text.split("\n"))


def test_fenced_block_becomes_a_code_block_directive() -> None:
    out = _convert("Example:\n\n```python\nx = 1\n```\n")
    assert ".. code-block:: python" in out
    assert "```python" not in "\n".join(out)


def test_fenced_block_content_is_indented_under_the_directive() -> None:
    out = _convert("```python\nx = 1\n```")
    directive = next(i for i, line in enumerate(out) if ".. code-block::" in line)
    body = [line for line in out[directive:] if line.strip() == "x = 1"]
    assert body, "code body missing"
    assert body[0].startswith("   ")


def test_indented_fence_keeps_its_indentation() -> None:
    out = _convert("Example:\n\n    ```python\n    x = 1\n    ```\n")
    assert any(line.startswith("    .. code-block:: python") for line in out)


def test_fence_without_a_language_still_converts() -> None:
    out = _convert("```\nplain\n```")
    assert any(".. code-block::" in line for line in out)
    assert "```" not in "\n".join(out)


def test_inline_single_backticks_become_double() -> None:
    assert "``ENDPOINT``" in "\n".join(_convert("Reads the `ENDPOINT` env-var."))


def test_existing_double_backticks_are_left_alone() -> None:
    out = "\n".join(_convert("Already ``fine`` here."))
    assert "``fine``" in out
    assert "````" not in out


def test_backticks_inside_a_code_block_are_left_alone() -> None:
    out = "\n".join(_convert("```python\na = `b`\n```"))
    assert "a = `b`" in out


def test_plain_prose_is_unchanged() -> None:
    assert _convert("Just prose.") == ["Just prose."]


def test_bold_is_left_alone_because_rst_shares_the_syntax() -> None:
    assert "**Automatic Checkpointing**" in "\n".join(
        _convert("- **Automatic Checkpointing**: saves progress")
    )


def test_blank_line_is_inserted_before_the_directive() -> None:
    """RST requires a blank line before a directive; Markdown does not."""
    out = _convert("**Persistent Deployment:**\n```python\nx = 1\n```")
    directive = next(i for i, line in enumerate(out) if ".. code-block::" in line)
    assert out[directive - 1] == "", "directive must be preceded by a blank line"


def test_no_duplicate_blank_line_when_one_already_precedes() -> None:
    out = _convert("Example:\n\n```python\nx = 1\n```")
    directive = next(i for i, line in enumerate(out) if ".. code-block::" in line)
    assert out[directive - 1] == ""
    assert out[directive - 2] != "" or out[directive - 2] == "Example:"
