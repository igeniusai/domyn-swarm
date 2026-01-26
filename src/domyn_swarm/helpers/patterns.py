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

from __future__ import annotations

import re

_BRACE_RE = re.compile(r"\{([^{}]+)\}")
_RANGE_RE = re.compile(r"^\s*(-?\d+)\s*\.\.\s*(-?\d+)\s*(?:\.\.\s*(-?\d+)\s*)?$")


def expand_brace_ranges(pattern: str, *, max_expansions: int = 10_000) -> list[str]:
    """Expand bash-like numeric brace ranges in a path pattern.

    This supports numeric brace ranges like:
    - `file-{1..3}.parquet` -> `file-1.parquet`, `file-2.parquet`, `file-3.parquet`
    - `file-{0001..0003}.parquet` -> zero-padded expansion preserving width
    - `file-{1..5..2}.parquet` -> supports optional step (`1,3,5`)

    Non-numeric braces are left untouched so literal braces in filenames continue to work.
    If multiple brace ranges are present, expansion is applied left-to-right.

    Args:
        pattern: String pattern potentially containing `{start..end[..step]}`.
        max_expansions: Safety bound on the total number of expanded patterns.

    Returns:
        List of expanded patterns. If no supported brace range is found, returns `[pattern]`.

    Raises:
        ValueError: If expansion would exceed `max_expansions` or a numeric range is invalid.
    """
    out: list[str] = [pattern]

    while True:
        expanded_any = False
        next_out: list[str] = []

        for item in out:
            match = _BRACE_RE.search(item)
            if match is None:
                next_out.append(item)
                continue

            inner = match.group(1)
            range_match = _RANGE_RE.match(inner)
            if range_match is None:
                # Not a supported numeric range; leave braces as-is.
                next_out.append(item)
                continue

            expanded_any = True
            start_s, end_s, step_s = range_match.groups()
            assert start_s is not None and end_s is not None
            start = int(start_s)
            end = int(end_s)

            if step_s is None:
                step = 1 if start <= end else -1
            else:
                step = int(step_s)
                if step == 0:
                    raise ValueError("Brace expansion step must not be 0.")

            # Use max width like bash does; this preserves leading zero padding.
            width = max(len(start_s.lstrip("-")), len(end_s.lstrip("-")))

            seq = range(start, end + 1, step) if step > 0 else range(start, end - 1, step)

            prefix = item[: match.start()]
            suffix = item[match.end() :]
            for n in seq:
                sign = "-" if n < 0 else ""
                num = f"{abs(n):0{width}d}"
                next_out.append(prefix + sign + num + suffix)

            if len(next_out) > max_expansions:
                raise ValueError(
                    f"Brace expansion produced too many patterns (> {max_expansions})."
                )

        out = next_out
        if not expanded_any:
            return out
