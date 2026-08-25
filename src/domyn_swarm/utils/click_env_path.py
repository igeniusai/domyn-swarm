# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import pathlib

import click


class ClickEnvPath(click.ParamType):
    """Click parameter type that expands environment variables in paths."""

    name = "envpath"

    def convert(self, value, param, ctx):
        """Convert a CLI value into a pathlib path."""
        from domyn_swarm.utils.env_path import EnvPath

        try:
            return pathlib.Path(str(EnvPath(value)))
        except Exception as e:
            self.fail(f"{value!r} is not a valid path: {e}", param, ctx)
