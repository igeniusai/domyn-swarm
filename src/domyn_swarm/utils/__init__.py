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

from typing import TYPE_CHECKING

__all__ = [
    "ClickEnvPath",
    "EnvPath",
]


def __getattr__(name: str):
    """Load utility classes on first access."""
    if name == "ClickEnvPath":
        from domyn_swarm.utils.click_env_path import ClickEnvPath

        return ClickEnvPath
    if name == "EnvPath":
        from domyn_swarm.utils.env_path import EnvPath

        return EnvPath
    raise AttributeError(name)


if TYPE_CHECKING:
    from domyn_swarm.utils.click_env_path import ClickEnvPath
    from domyn_swarm.utils.env_path import EnvPath
