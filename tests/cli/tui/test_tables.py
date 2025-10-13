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

from rich.style import Style
from rich.table import Table

from domyn_swarm.cli.tui import tables as T


def _style_equals(style_obj, expected: str) -> bool:
    """Normalize style comparisons (Rich may store as str or Style)."""
    if style_obj is None:
        return expected == ""
    return Style.parse(str(style_obj)) == Style.parse(expected)


def test_kv_table_structure():
    tbl = T._kv_table()
    assert isinstance(tbl, Table)

    # Columns
    assert len(tbl.columns) == 2

    field_col = tbl.columns[0]
    value_col = tbl.columns[1]

    # First column: name/style/wrap/justify
    assert field_col.header == "Field"
    assert _style_equals(field_col.style, "bold dim")
    assert field_col.no_wrap is True
    assert field_col.justify == "right"

    # Second column: name & overflow folding
    assert value_col.header == "Value"
    # Overflow may be stored as a literal or enum-like; string-compare is fine
    assert str(value_col.overflow) == "fold"


def test_list_table_config_and_columns():
    cols = [" Name", "Backend", "Phase", "Endpoint", "Notes"]
    tbl = T.list_table(columns=cols)
    assert isinstance(tbl, Table)

    # Table-level config
    assert tbl.show_header is True
    assert _style_equals(tbl.header_style, "bold")
    assert tbl.expand is True
    assert tbl.pad_edge is False

    # Columns presence & order
    assert [c.header for c in tbl.columns] == cols
    for c in tbl.columns:
        assert str(c.overflow) == "fold"


def test_list_table_with_no_columns_ok():
    tbl = T.list_table(columns=[])
    assert isinstance(tbl, Table)
    assert tbl.show_header is True
    # No columns added
    assert len(tbl.columns) == 0
