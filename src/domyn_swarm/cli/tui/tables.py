# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from rich.table import Table


def _kv_table() -> Table:
    """Create a key-value table with grid layout for displaying field-value pairs.

    Creates a Rich Table with grid layout optimized for displaying key-value pairs.
    The table has two columns: a right-justified "Field" column with bold dim styling
    and no wrapping, and a "Value" column that allows text overflow folding.

    Returns:
        Table: A configured Rich Table object with grid layout and two columns
            for displaying key-value pairs with appropriate styling and formatting.
    """
    t = Table.grid(padding=(0, 1))
    t.add_column("Field", style="bold dim", no_wrap=True, justify="right")
    t.add_column("Value", overflow="fold")
    return t


def list_table(*, columns: list[str]) -> Table:
    t = Table(show_header=True, header_style="bold", expand=True, pad_edge=False)
    for c in columns:
        t.add_column(c, overflow="fold")
    return t
