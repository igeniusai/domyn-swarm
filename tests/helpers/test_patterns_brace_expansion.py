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

import pytest

from domyn_swarm.helpers.patterns import expand_brace_ranges


def test_expand_brace_ranges_preserves_zero_padding() -> None:
    out = expand_brace_ranges("file-00{0978..0980}.parquet")
    assert out == [
        "file-000978.parquet",
        "file-000979.parquet",
        "file-000980.parquet",
    ]


def test_expand_brace_ranges_supports_step() -> None:
    out = expand_brace_ranges("file-{1..5..2}.parquet")
    assert out == ["file-1.parquet", "file-3.parquet", "file-5.parquet"]


def test_expand_brace_ranges_multiple_ranges_cross_product() -> None:
    out = expand_brace_ranges("a{1..2}-b{3..4}.parquet")
    assert out == [
        "a1-b3.parquet",
        "a1-b4.parquet",
        "a2-b3.parquet",
        "a2-b4.parquet",
    ]


def test_expand_brace_ranges_leaves_non_numeric_braces_untouched() -> None:
    out = expand_brace_ranges("file-{abc}.parquet")
    assert out == ["file-{abc}.parquet"]


def test_expand_brace_ranges_refuses_too_many_expansions() -> None:
    with pytest.raises(ValueError, match="too many"):
        _ = expand_brace_ranges("x{1..20000}.parquet", max_expansions=1000)
