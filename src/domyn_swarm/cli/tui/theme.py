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
