# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""`JobConfig` — the single statement of a job's configuration."""

import json

import pytest

from domyn_swarm.jobs.api.base import SwarmJob
from domyn_swarm.jobs.api.builder import JobBuilder
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


def test_assigning_output_cols_rederives_default_output_cols() -> None:
    """Assignment re-derives the derived value, as construction does.

    `SwarmJob` routes attribute writes into the config, and several job classes
    assign `output_cols` after `super().__init__()`. If the derivation did not
    follow the write, `default_output_cols` -- what the runner writes the
    results under when the caller names no columns -- would keep describing the
    columns the job no longer produces.
    """
    cfg = JobConfig(model="m")
    cfg.output_cols = ["a", "b"]
    assert cfg.default_output_cols == ["a", "b"]


def test_assigning_output_cols_preserves_an_explicit_default_output_cols() -> None:
    """A value the caller chose is never re-derived away."""
    cfg = JobConfig(model="m", default_output_cols=["x"])
    cfg.output_cols = ["a", "b"]
    assert cfg.default_output_cols == ["x"]


def test_a_reconstructed_config_still_rederives_default_output_cols() -> None:
    """A derived value stays derived across the trip through `--job-kwargs`.

    `model_validate(model_dump())` marks every field as explicitly set, so a
    rebuilt config would carry its merely *derived* `default_output_cols` as if
    a caller had chosen it, and the next override of `output_cols` would leave
    it stale. Dumping with `exclude_unset` keeps set-ness the same on both
    sides of the wire.
    """
    cfg = JobConfig(model="m", output_cols="answer")
    rebuilt = JobConfig.model_validate(cfg.model_dump(exclude_unset=True))
    assert rebuilt.default_output_cols == ["answer"]

    assert rebuilt.merged_with(output_cols="other").default_output_cols == ["other"]

    explicit = JobConfig(model="m", output_cols="answer", default_output_cols=["x"])
    rebuilt_explicit = JobConfig.model_validate(explicit.model_dump(exclude_unset=True))
    assert rebuilt_explicit.merged_with(output_cols="other").default_output_cols == ["x"]


def test_subclass_instance_state_is_not_serialized(monkeypatch) -> None:
    """Subclass instance state stays out of the `--job-kwargs` payload.

    Serializing it would send it to the subprocess as an unrecognised kwarg,
    which is then forwarded to the provider as a request parameter.
    """
    monkeypatch.setenv("ENDPOINT", "http://dummy-endpoint")

    class J(SwarmJob):
        async def transform_items(self, items: list):
            return items

        def __init__(self, **kw):
            super().__init__(**kw)
            self.tokenizer_path = "/scratch/tok"
            self.rows_seen = 0

    payload = json.loads(JobBuilder.to_kwargs_json(J(model="m")))
    assert "tokenizer_path" not in payload
    assert "rows_seen" not in payload


def test_a_job_writing_output_cols_serializes_matching_default_output_cols(
    monkeypatch,
) -> None:
    """A job that resolves its own output columns after `super().__init__()`
    sends those columns over the wire, and the rebuilt job derives the same
    defaults from them."""
    monkeypatch.setenv("ENDPOINT", "http://dummy-endpoint")

    class J(SwarmJob):
        async def transform_items(self, items: list):
            return items

        def __init__(self, **kw):
            super().__init__(**kw)
            self.output_cols = ["a", "b"]

    job = J(model="m")
    payload = json.loads(JobBuilder.to_kwargs_json(job))
    assert "default_output_cols" not in payload, "a derived value is not serialized"

    rebuilt = J(model="m", **payload)
    assert rebuilt.output_cols == ["a", "b"]
    assert rebuilt.default_output_cols == job.default_output_cols == ["a", "b"]


def test_request_params_mutated_in_place_are_serialized(monkeypatch) -> None:
    """A provider parameter a job adds to `job.kwargs` reaches the subprocess.

    Mutating the dict is not an assignment, so it never marks `request_params`
    as set. Serializing with `exclude_unset` would otherwise drop it, and the
    job would behave differently on the cluster than it does locally.
    """
    monkeypatch.setenv("ENDPOINT", "http://dummy-endpoint")

    class J(SwarmJob):
        async def transform_items(self, items: list):
            return items

        def __init__(self, **kw):
            super().__init__(**kw)
            self.kwargs["temperature"] = 0.0

    job = J(model="m")
    payload = json.loads(JobBuilder.to_kwargs_json(job))
    assert payload["request_params"] == {"temperature": 0.0}
    assert J(model="m", **payload)._request_kwargs() == {"temperature": 0.0}


def test_a_caller_supplied_config_is_not_mutated(monkeypatch) -> None:
    """Building a job leaves the config it was given alone.

    `__init__` resolves `name` and `endpoint` into the config, and a job may
    write to it afterwards, so it works on a copy -- a deep one, since
    `request_params` is mutated in place.
    """
    monkeypatch.setenv("ENDPOINT", "http://dummy-endpoint")

    class J(SwarmJob):
        async def transform_items(self, items: list):
            return items

        def __init__(self, **kw):
            super().__init__(**kw)
            self.output_cols = "answer"
            self.kwargs["temperature"] = 0.0

    cfg = JobConfig(model="m")
    job = J(config=cfg)

    assert job.name == "J"
    assert cfg.name is None
    assert cfg.endpoint is None
    assert cfg.output_cols == "result"
    assert cfg.request_params == {}


class _ConcreteJob(SwarmJob):
    """A minimal concrete `SwarmJob` for exercising attribute routing directly."""

    async def transform_items(self, items: list):
        return items


def test_unknown_attribute_on_a_constructed_job_raises_attribute_error(monkeypatch) -> None:
    """A name that is neither instance state nor a config field is a genuine
    `AttributeError`, not an infinite recursion into `__getattr__`."""
    monkeypatch.setenv("ENDPOINT", "http://dummy-endpoint")
    job = _ConcreteJob(model="m")
    with pytest.raises(AttributeError):
        _ = job.definitely_not_a_field


def test_getattr_on_a_bare_instance_does_not_recurse() -> None:
    """`__getattr__` must not recurse before `config` has been assigned.

    `Job.__new__(Job)` produces an instance with no `config` in `__dict__` yet --
    the window between allocation and `__init__` (and where the copy/pickle
    protocols probe an instance, see below). `__getattr__` reads
    `self.__dict__.get("config")` rather than `self.config`; the latter would
    itself miss and re-enter `__getattr__`, recursing until `RecursionError`.
    """
    raw = _ConcreteJob.__new__(_ConcreteJob)
    with pytest.raises(AttributeError):
        _ = raw.definitely_not_a_field


def test_copy_protocol_probe_on_a_bare_instance_returns_false() -> None:
    """A `hasattr(obj, "__dunder__")` probe -- as `copy`/`pickle` perform -- must
    resolve to `False`, not raise `RecursionError` or anything else."""
    raw = _ConcreteJob.__new__(_ConcreteJob)
    assert hasattr(raw, "__deepcopy__") is False


def test_a_subclass_property_named_after_a_config_field_gets_writes(monkeypatch) -> None:
    """A write to a property whose name shadows a config field reaches the setter.

    A read of `retries` on such a subclass hits the property, since a data
    descriptor beats `__getattr__` in normal attribute lookup. Without the
    matching check in `__setattr__`, a write would skip the property and land in
    the config, so the setter would never run.
    """
    monkeypatch.setenv("ENDPOINT", "http://dummy-endpoint")
    calls = []

    class J(SwarmJob):
        async def transform_items(self, items: list):
            return items

        @property
        def retries(self):
            return self.config.retries

        @retries.setter
        def retries(self, value):
            calls.append(value)
            self.config.retries = value

    job = J(model="m")
    job.retries = 9
    assert calls == [9], "the property setter must run"
    assert job.retries == 9


class _Plain(SwarmJob):
    async def transform_items(self, items: list):
        return items


def test_unknown_kwargs_are_rejected(monkeypatch) -> None:
    """Bare provider parameters warned through 0.33 and are now an error."""
    monkeypatch.setenv("ENDPOINT", "http://dummy-endpoint")
    with pytest.raises(TypeError, match=r"request_params=\{\.\.\.\}"):
        _Plain(model="m", temperature=0.2)


def test_explicit_request_params_do_not_warn(monkeypatch, recwarn) -> None:
    """The supported form is silent."""
    monkeypatch.setenv("ENDPOINT", "http://dummy-endpoint")
    job = _Plain(model="m", request_params={"temperature": 0.2})
    assert job._request_kwargs() == {"temperature": 0.2}
    assert [w for w in recwarn if issubclass(w.category, DeprecationWarning)] == []


def test_nested_kwargs_form_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("ENDPOINT", "http://dummy-endpoint")
    with pytest.raises(TypeError, match="'kwargs' is not a configuration field"):
        _Plain(model="m", kwargs={"temperature": 0.2})


def test_a_near_miss_of_a_field_name_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("ENDPOINT", "http://dummy-endpoint")
    with pytest.raises(TypeError, match="did you mean 'retries'"):
        _Plain(model="m", retres=3)


def test_a_name_resembling_nothing_points_to_request_params(monkeypatch) -> None:
    """Only near misses get a field suggestion; other names point to `request_params`."""
    monkeypatch.setenv("ENDPOINT", "http://dummy-endpoint")
    with pytest.raises(TypeError) as excinfo:
        _Plain(model="m", frequency_penalty=0.4)
    assert "did you mean" not in str(excinfo.value)
    assert "request_params={...}" in str(excinfo.value)


def test_a_provider_param_named_like_a_config_field_is_expressible(monkeypatch) -> None:
    """A provider parameter may share a name with a configuration field.

    The two are set independently and neither shadows the other.
    """
    monkeypatch.setenv("ENDPOINT", "http://dummy-endpoint")
    job = _Plain(model="m", max_concurrency=4, request_params={"max_concurrency": 5})
    assert job.max_concurrency == 4
    assert job._request_kwargs() == {"max_concurrency": 5}


def test_output_column_name_is_still_deprecated_and_still_works(monkeypatch) -> None:
    """`output_column_name` still warns, still sets `output_cols`."""
    monkeypatch.setenv("ENDPOINT", "http://dummy-endpoint")
    with pytest.warns(DeprecationWarning, match="output_column_name"):
        job = _Plain(model="m", output_column_name="answer")
    assert job.output_cols == "answer"
    assert job._request_kwargs() == {}, "it must not leak to the provider"


def test_an_everyday_typo_just_below_a_stricter_cutoff_is_rejected(monkeypatch) -> None:
    """`retrys` is an everyday typo of `retries`, at resemblance ratio 0.769.

    It is why the cutoff is 0.75 and not 0.8: above 0.769 the error would not
    suggest `retries`.
    """
    monkeypatch.setenv("ENDPOINT", "http://dummy-endpoint")
    with pytest.raises(TypeError, match="did you mean 'retries'"):
        _Plain(model="m", retrys=3)


def test_a_provider_param_sharing_a_prefix_with_a_field_is_not_called_a_typo(
    monkeypatch,
) -> None:
    """The cutoff must not report real provider parameters as typos.

    `max_tokens` is a common provider parameter sharing a `max_` prefix with
    the `max_concurrency` field; at ratio 0.56 it is well below the 0.75
    suggestion threshold.
    """
    monkeypatch.setenv("ENDPOINT", "http://dummy-endpoint")
    with pytest.raises(TypeError) as excinfo:
        _Plain(model="m", max_tokens=256)
    assert "did you mean" not in str(excinfo.value)
