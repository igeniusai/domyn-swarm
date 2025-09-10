import pathlib

import click

from domyn_swarm import utils


class ClickEnvPath(click.ParamType):
    name = "envpath"

    def convert(self, value, param, ctx):
        try:
            return pathlib.Path(str(utils.EnvPath(value)))
        except Exception as e:
            self.fail(f"{value!r} is not a valid path: {e}", param, ctx)
