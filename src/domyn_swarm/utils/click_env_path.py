import pathlib

import click

import domyn_swarm


class ClickEnvPath(click.ParamType):
    name = "envpath"

    def convert(self, value, param, ctx):
        try:
            return pathlib.Path(str(domyn_swarm.utils.EnvPath(value)))
        except Exception as e:
            self.fail(f"{value!r} is not a valid path: {e}", param, ctx)
