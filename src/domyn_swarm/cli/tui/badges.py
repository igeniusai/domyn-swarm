# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

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
