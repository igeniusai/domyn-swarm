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
