# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Class-level configuration declarations."""

import pytest

from domyn_swarm.jobs.api.base import SwarmJob
from domyn_swarm.jobs.api.config import JobConfig


@pytest.fixture(autouse=True)
def _endpoint(monkeypatch):
    monkeypatch.setenv("ENDPOINT", "http://dummy-endpoint")


def test_bare_attributes_take_effect() -> None:
    """A declared field is applied."""
    with pytest.warns(UserWarning, match="max_concurrency"):

        class J(SwarmJob):
            max_concurrency = 32
            retries = 9

            async def transform_items(self, items: list):
                return items

    job = J(model="m")
    assert job.max_concurrency == 32
    assert job.retries == 9


def test_a_declaration_equal_to_the_default_does_not_warn(recwarn) -> None:
    """Declaring the default changes nothing, so it is not worth a warning."""

    class J(SwarmJob):
        max_concurrency = 2  # the field default

        async def transform_items(self, items: list):
            return items

    assert [w for w in recwarn if issubclass(w.category, UserWarning)] == []
    assert J(model="m").max_concurrency == 2


def test_constructor_argument_beats_the_declaration() -> None:
    """Explicit arguments win over a class-level declaration."""
    with pytest.warns(UserWarning):

        class J(SwarmJob):
            retries = 9

            async def transform_items(self, items: list):
                return items

    assert J(model="m", retries=1).retries == 1


def test_config_object_declaration() -> None:
    """A class-level `config` object is used as the base configuration."""

    class J(SwarmJob):
        config = JobConfig(model="declared", max_concurrency=7)

        async def transform_items(self, items: list):
            return items

    job = J()
    assert job.model == "declared"
    assert job.max_concurrency == 7


def test_declaring_both_forms_for_one_field_raises() -> None:
    """A bare attribute contradicting the config object is an error."""
    with pytest.raises(TypeError, match="retries"):

        class J(SwarmJob):
            config = JobConfig(model="m", retries=3)
            retries = 9

            async def transform_items(self, items: list):
                return items


def test_api_version_is_not_swept_up() -> None:
    """`api_version` is a class marker, not configuration."""

    class J(SwarmJob):
        api_version = 2

        async def transform_items(self, items: list):
            return items

    assert J.api_version == 2
    assert "api_version" not in JobConfig.model_fields


def test_descendant_bare_attribute_composes_with_an_ancestor_config_declaration() -> None:
    """A subclass's bare attribute is not dropped when an ancestor uses `config=`.

    The MRO walk must fold in every ancestor's declarations rather than stopping
    at the first `config` object it finds, or a descendant's own declaration is
    silently ignored while the class-definition warning claims it took effect.
    """
    with pytest.warns(UserWarning, match="retries"):

        class Parent(SwarmJob):
            config = JobConfig(model="declared", max_concurrency=7)

            async def transform_items(self, items: list):
                return items

        class Child(Parent):
            retries = 99

            async def transform_items(self, items: list):
                return items

    job = Child(model="m")
    assert job.retries == 99
    assert job.max_concurrency == 7, "the ancestor's config field is still inherited"
    assert job.model == "m"


def test_descendant_bare_attribute_overrides_a_field_the_ancestor_config_set() -> None:
    """A descendant's declaration for the *same* field wins over the ancestor's
    `config` object."""
    with pytest.warns(UserWarning, match="max_concurrency"):

        class Parent(SwarmJob):
            config = JobConfig(model="declared", max_concurrency=7)

            async def transform_items(self, items: list):
                return items

        class Child(Parent):
            max_concurrency = 64

            async def transform_items(self, items: list):
                return items

    job = Child(model="m")
    assert job.max_concurrency == 64


def test_three_levels_mixing_config_object_and_bare_attributes() -> None:
    """Declarations compose across three levels, each layer winning over the last
    for the fields it declares."""
    with pytest.warns(UserWarning):

        class Grandparent(SwarmJob):
            config = JobConfig(model="declared", max_concurrency=7, timeout=100)

            async def transform_items(self, items: list):
                return items

        class Parent(Grandparent):
            retries = 3

            async def transform_items(self, items: list):
                return items

        class Child(Parent):
            timeout = 5

            async def transform_items(self, items: list):
                return items

    job = Child(model="m")
    assert job.max_concurrency == 7, "untouched field from the grandparent's config"
    assert job.retries == 3, "declared by the parent"
    assert job.timeout == 5, "the child overrides the grandparent's config value"


def test_config_and_a_non_overlapping_bare_attribute_on_one_class_is_allowed() -> None:
    """`config=` for some fields and a bare attribute for another field on the same
    class is not a contradiction -- only declaring the *same* field twice is."""

    class J(SwarmJob):
        config = JobConfig(model="m")
        max_concurrency = 32

        async def transform_items(self, items: list):
            return items

    job = J()
    assert job.model == "m"
    assert job.max_concurrency == 32


def test_declaring_both_forms_for_the_same_field_names_only_that_field() -> None:
    """The conflict message names only the field that conflicts, not every bare
    attribute on the class."""
    with pytest.raises(TypeError) as exc_info:

        class J(SwarmJob):
            config = JobConfig(model="m", retries=3)
            retries = 9
            max_concurrency = 32

            async def transform_items(self, items: list):
                return items

    message = str(exc_info.value)
    assert "retries" in message
    assert "max_concurrency" not in message
