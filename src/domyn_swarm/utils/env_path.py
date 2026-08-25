# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib

from pydantic_core import core_schema

# This is a workaround for the fact that pathlib.Path is a built-in class
# and cannot be directly inherited from.
# Using python 3.10, inheriting from os.PathLike raises AttributeError:
# type object 'EnvPath' has no attribute '_flavour'
# This way we create a new class that inherits from the same base class as pathlib.Path
# and we can then extend it.
_EnvBase = type(pathlib.Path())


def _expand(s: os.PathLike | str) -> str:
    s = os.path.expandvars(os.fspath(s))
    s = os.path.expanduser(s)
    return s


class EnvPath(_EnvBase):
    def __new__(cls, *args):
        parts = tuple(_expand(a) for a in args) if args else ("",)
        return super().__new__(cls, *parts)

    def joinpath(self, *args):
        return super().joinpath(*(_expand(a) for a in args))

    def __truediv__(self, other):
        return self.joinpath(other)

    def __str__(self) -> str:
        return _expand(super().__str__())

    def __fspath__(self) -> str:
        return self.__str__()

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return core_schema.no_info_after_validator_function(
            function=lambda v: cls(v),  # construct EnvPath
            schema=core_schema.union_schema(
                [
                    core_schema.str_schema(),
                    core_schema.is_instance_schema(pathlib.Path),
                    core_schema.is_instance_schema(os.PathLike),
                ]
            ),
            serialization=core_schema.to_string_ser_schema(),
        )
