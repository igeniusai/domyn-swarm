# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import os

_PHASE_GLYPH = {
    "RUNNING": "●",
    "INITIALIZING": "◔",
    "PENDING": "⋯",
    "FAILED": "✖",
    "STOPPED": "■",
    "UNKNOWN": "?",
}

# Optional ASCII fallback (set DOMYN_SWARM_ASCII=1 to enable)
_ASCII_GLYPH = {
    "RUNNING": "*",
    "INITIALIZING": "~",
    "PENDING": "...",
    "FAILED": "x",
    "STOPPED": "#",
    "UNKNOWN": "?",
}

PHASE_STYLE = {
    "RUNNING": "bold white on green3",
    "INITIALIZING": "bold black on yellow3",
    "PENDING": "bold black on khaki1",
    "FAILED": "bold white on red3",
    "STOPPED": "bold white on grey39",
    "UNKNOWN": "bold white on grey23",
}

_BAD_STATES = {"FAILED", "CANCELLED", "TIMEOUT", "BOOT_FAIL", "NODE_FAIL"}
_WAIT_STATES = {"PENDING", "CONFIGURING", "CREATING", "STARTING", "INITIALIZING"}


def phase_style(s: str) -> str:
    return PHASE_STYLE.get(s.upper(), PHASE_STYLE["UNKNOWN"])


def phase_glyph(s: str) -> str:
    if os.getenv("DOMYN_SWARM_ASCII", "").strip() == "1":
        return _ASCII_GLYPH.get(s.upper(), "?")
    return _PHASE_GLYPH.get(s.upper(), "?")
