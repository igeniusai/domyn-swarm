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

import re

from ulid import ULID

# Backend limits
_MAX_NAME = {"lepton": 36, "slurm": 63}
_SAFE_DEFAULT_LIMIT = min(_MAX_NAME.values())


def _slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "swarm"


def generate_swarm_name(name: str, backend: str) -> str:
    """Generate a unique swarm name by appending a short UUID to the given name."""
    limit = _MAX_NAME.get(backend.lower(), _SAFE_DEFAULT_LIMIT)
    suffix = str(ULID())[:10].lower()
    max_slug_len = max(1, limit - 1 - len(suffix))
    slug = _slugify(name)[:max_slug_len]
    return f"{slug}-{suffix}"
