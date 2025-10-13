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

from rich.text import Text

from .theme import phase_glyph, phase_style


def phase_badge(phase: str) -> Text:
    t = Text(f"{phase_glyph(phase)} {phase}")
    t.stylize(phase_style(phase))
    return t


def http_badge(http: int | str | None):
    if http is None:
        return None
    if http == 200:
        txt = Text("200 OK")
        txt.stylize("bold green")
        return txt
    s = str(http).lower()
    if s in {"unready", "timeout"}:
        txt = Text(s)
        txt.stylize("bold yellow")
        return txt
    txt = Text(str(http))
    txt.stylize("bold red")
    return txt
