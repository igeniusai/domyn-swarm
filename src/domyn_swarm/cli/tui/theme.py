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
_PHASE_STYLE = {
    "RUNNING": "bold white on green3",
    "INITIALIZING": "bold black on yellow3",
    "PENDING": "bold black on khaki1",
    "FAILED": "bold white on red3",
    "STOPPED": "bold white on grey39",
    "UNKNOWN": "bold white on dark_sea_green4",
}
_PHASE_EMOJI = {
    "RUNNING": "✅",
    "INITIALIZING": "🟡",
    "PENDING": "⏳",
    "FAILED": "❌",
    "STOPPED": "⏹️",
    "UNKNOWN": "❔",
}

_BAD_STATES = {"FAILED", "CANCELLED", "TIMEOUT", "BOOT_FAIL", "NODE_FAIL"}
_WAIT_STATES = {"PENDING", "CONFIGURING", "CREATING", "STARTING", "INITIALIZING"}


def phase_style(phase: str) -> str:
    return _PHASE_STYLE.get(phase, "bold white on grey23")


def phase_emoji(phase: str) -> str:
    return _PHASE_EMOJI.get(phase, "•")
