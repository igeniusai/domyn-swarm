# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""`JobConfig` — the single statement of a job's configuration."""

import pytest

from domyn_swarm.jobs.api.config import JobConfig, OutputJoinMode


def test_defaults_match_the_constructor() -> None:
    """Field defaults reproduce `SwarmJob.__init__`'s effective defaults."""
    cfg = JobConfig(model="m")
    assert cfg.provider == "openai"
    assert cfg.input_column_name == "messages"
    assert cfg.id_column_name is None
    assert cfg.output_cols == "result"
    assert cfg.default_output_cols == ["result"]
    assert cfg.output_mode is OutputJoinMode.APPEND
    assert cfg.checkpoint_interval == 16
    assert cfg.max_concurrency == 2
    assert cfg.retries == 5
    assert cfg.timeout == 600
    assert cfg.request_params == {}


def test_default_output_cols_is_derived() -> None:
    """A string `output_cols` yields a one-element `default_output_cols`."""
    assert JobConfig(model="m", output_cols="answer").default_output_cols == ["answer"]
    assert JobConfig(model="m", output_cols=["a", "b"]).default_output_cols == ["a", "b"]


def test_explicit_default_output_cols_is_not_overwritten() -> None:
    """Deriving does not clobber an explicit value."""
    cfg = JobConfig(model="m", output_cols="answer", default_output_cols=["x"])
    assert cfg.default_output_cols == ["x"]


def test_unknown_field_is_rejected() -> None:
    """An unknown name is rejected because it is unknown."""
    with pytest.raises(ValueError, match="temperatur"):
        JobConfig(model="m", temperatur=0.2)


def test_a_config_name_is_not_reported_as_a_misspelling_of_itself() -> None:
    """An exact field name binds to its field, never to a near-miss suggestion."""
    assert JobConfig(model="m", max_concurrency=5).max_concurrency == 5


def test_request_params_may_collide_with_a_config_name() -> None:
    """A provider parameter named like a config field is expressible.

    Passthrough has its own namespace, so the two never compete.
    """
    cfg = JobConfig(model="m", timeout=30, request_params={"timeout": 5, "model": "other"})
    assert cfg.timeout == 30
    assert cfg.request_params == {"timeout": 5, "model": "other"}


def test_merged_with_overrides_fields() -> None:
    """`merged_with` returns a new config with the overrides applied."""
    cfg = JobConfig(model="m", retries=2)
    merged = cfg.merged_with(retries=9, max_concurrency=4)
    assert (merged.retries, merged.max_concurrency) == (9, 4)
    assert cfg.retries == 2, "the original is not mutated"


def test_merged_with_rederives_default_output_cols_from_a_new_output_cols() -> None:
    """Overriding `output_cols` alone re-derives `default_output_cols`.

    A merely *derived* `default_output_cols` must not travel through
    `merged_with`, or the new config's validator would see a non-`None` value
    and leave it describing the columns the job no longer produces.
    """
    cfg = JobConfig(model="m")  # output_cols="result" (default, not explicit)
    merged = cfg.merged_with(output_cols="answer")
    assert merged.output_cols == "answer"
    assert merged.default_output_cols == ["answer"]


def test_merged_with_preserves_an_explicit_default_output_cols() -> None:
    """An explicitly-set `default_output_cols` survives `merged_with`.

    The half `exclude_unset` could plausibly break: a genuinely explicit value
    must still travel through the dump, not just a derived one being dropped.
    """
    cfg = JobConfig(model="m", output_cols="answer", default_output_cols=["x"])
    merged = cfg.merged_with(retries=9)
    assert merged.default_output_cols == ["x"]


def test_output_cols_none_resolves_to_result() -> None:
    """An explicit `output_cols=None` resolves to `"result"`.

    A subclass forwarding `**kwargs` that happens to carry
    `output_cols=None` must keep working.
    """
    cfg = JobConfig(model="m", output_cols=None)
    assert cfg.output_cols == "result"
    assert cfg.default_output_cols == ["result"]


def test_assignment_is_validated() -> None:
    """`validate_assignment` keeps a mutated config well-typed."""
    cfg = JobConfig(model="m")
    cfg.output_cols = ["a", "b"]
    assert cfg.output_cols == ["a", "b"]
    with pytest.raises(ValueError):
        cfg.max_concurrency = "not an int"
