import pydantic_core
import pathlib
import os


class EnvPath(pathlib.Path):
    _flavour = type(pathlib.Path())._flavour  # Required for subclassing Path

    def __new__(cls, *args, **kwargs):
        raw_path = os.path.expandvars(os.path.join(*map(str, args)))
        return super().__new__(cls, raw_path, **kwargs)

    def __str__(self):
        return super().__str__()

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return pydantic_core.core_schema.no_info_after_validator_function(
            function=cls,
            schema=pydantic_core.core_schema.str_schema(),
            serialization=pydantic_core.core_schema.plain_serializer_function_ser_schema(
                lambda v: str(v),
                when_used="always",
            ),
        )
