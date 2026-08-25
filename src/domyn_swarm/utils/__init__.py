# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

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
