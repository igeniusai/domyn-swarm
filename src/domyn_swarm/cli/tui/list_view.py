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

from collections.abc import Iterable

from rich.console import Console
from rich.padding import Padding
from rich.text import Text

from .badges import http_badge, phase_badge
from .tables import list_table


def render_swarm_list(rows: Iterable[object], *, console: Console):
    """
    Render a compact, responsive table of swarms.
    Each row must have: name, backend, phase, url, http, extra
    """
    t = list_table(columns=[" Name", "Backend", "Phase", "Endpoint", "Notes"])

    for r in rows:
        # pad 1 space left of the table border
        name = Padding(getattr(r, "name", "—"), (0, 0, 0, 1))
        backend = str(getattr(r, "backend", "—")).upper()
        phase = str(getattr(r, "phase", "UNKNOWN"))
        url = getattr(r, "url", None)
        http = getattr(r, "http", None)
        extra = getattr(r, "extra", None) or {}

        url_txt = Text(url or "—")
        if url:
            url_txt.stylize(f"link {url}")

        notes_parts = []
        hb = http_badge(http)
        if hb:
            notes_parts.append(str(hb))
        # small digest, not the whole dict
        notes_parts.extend([f"{k}={extra[k]}" for k in ("rep", "lb", "raw_state") if k in extra])
        notes = ", ".join(notes_parts) if notes_parts else "—"

        t.add_row(name, backend, phase_badge(phase), url_txt, notes)

    console.print(t)
