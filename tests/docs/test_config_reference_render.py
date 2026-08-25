# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the configuration-reference table renderer."""

from __future__ import annotations

import pathlib
from typing import Literal

from gen_config_reference import (
    env_var_name,
    render_annotation,
    render_default,
    render_model_table,
    render_settings_table,
)
from pydantic import BaseModel, Field


class _Sample(BaseModel):
    """A sample model."""

    required_field: str = Field(description="A required string.")
    optional_field: int = Field(default=7, description="An optional integer.")
    nullable_field: str | None = Field(default=None, description="Maybe a string.")
    kind: Literal["slurm", "lepton"] = Field(default="slurm", description="Which backend.")
    listed: list[str] = Field(default_factory=list, description="Some names.")
    piped: str = Field(default="a", description="Has a | pipe in the prose.")


def test_render_annotation_simple() -> None:
    assert render_annotation(str) == "str"
    assert render_annotation(int) == "int"


def test_render_annotation_union_uses_pipe() -> None:
    assert render_annotation(str | None) == "str | None"


def test_render_annotation_generic() -> None:
    assert render_annotation(list[str]) == "list[str]"


def test_render_annotation_literal_quotes_strings() -> None:
    assert render_annotation(Literal["slurm", "lepton"]) == '"slurm" | "lepton"'


def test_render_annotation_strips_module_paths() -> None:
    assert render_annotation(pathlib.Path) == "Path"


def test_render_default_marks_required() -> None:
    assert render_default(_Sample.model_fields["required_field"]) == "**required**"


def test_render_default_shows_value() -> None:
    assert render_default(_Sample.model_fields["optional_field"]) == "`7`"


def test_render_default_shows_none() -> None:
    assert render_default(_Sample.model_fields["nullable_field"]) == "`None`"


def test_render_default_marks_factory_as_computed() -> None:
    assert render_default(_Sample.model_fields["listed"]) == "*computed*"


def test_render_model_table_has_a_header_and_a_row_per_field() -> None:
    table = render_model_table(_Sample)
    assert "### `_Sample`" in table
    assert "| Field | Type | Default | Description |" in table
    for name in _Sample.model_fields:
        assert f"| `{name}` |" in table


def test_render_model_table_escapes_pipes_in_descriptions() -> None:
    table = render_model_table(_Sample)
    assert r"Has a \| pipe in the prose." in table


def test_render_model_table_includes_the_docstring() -> None:
    assert "A sample model." in render_model_table(_Sample)


class _SettingsLike(BaseModel):
    """Stands in for a BaseSettings model when testing env var naming."""

    model_config = {"env_prefix": "DOMYN_SWARM_"}

    log_level: str = Field(default="INFO", description="Global logging level.")
    defaults_file: str | None = Field(
        default=None, alias="DOMYN_SWARM_DEFAULTS", description="Defaults YAML path."
    )
    vllm_api_key: str | None = Field(
        default=None, alias="VLLM_API_KEY", description="Token used by vLLM."
    )


def test_env_var_name_applies_the_prefix() -> None:
    assert (
        env_var_name("log_level", _SettingsLike.model_fields["log_level"], "DOMYN_SWARM_")
        == "DOMYN_SWARM_LOG_LEVEL"
    )


def test_env_var_name_prefers_an_explicit_alias() -> None:
    field = _SettingsLike.model_fields["defaults_file"]
    assert env_var_name("defaults_file", field, "DOMYN_SWARM_") == "DOMYN_SWARM_DEFAULTS"


def test_env_var_name_alias_may_drop_the_prefix() -> None:
    field = _SettingsLike.model_fields["vllm_api_key"]
    assert env_var_name("vllm_api_key", field, "DOMYN_SWARM_") == "VLLM_API_KEY"


def test_render_settings_table_lists_env_var_names() -> None:
    table = render_settings_table(_SettingsLike)
    assert "| Variable | Type | Default | Description |" in table
    assert "| `DOMYN_SWARM_LOG_LEVEL` |" in table
    assert "| `DOMYN_SWARM_DEFAULTS` |" in table
    assert "| `VLLM_API_KEY` |" in table


def test_render_settings_table_uses_a_level_two_heading() -> None:
    assert render_settings_table(_SettingsLike).startswith("## ")
