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
