import json

import pytest

from domyn_swarm.jobs.api import JobBuilder
from domyn_swarm.jobs.api.base import SwarmJob


class DummyBuilderJob(SwarmJob):
    async def transform_items(self, items: list[object]) -> list[object]:
        return items


class NotAJob:
    pass


class InvalidKwargsJob(DummyBuilderJob):
    def to_kwargs(self):
        return ["not-a-dict"]


class UnserializableKwargsJob(DummyBuilderJob):
    def to_kwargs(self):
        return {"x": object()}


def test_from_class_path_builds_job_from_payload():
    class_path = f"{__name__}:DummyBuilderJob"
    kwargs_json = json.dumps({"name": "payload-name", "max_concurrency": 4})

    job = JobBuilder.from_class_path(
        class_path,
        kwargs_json,
        endpoint="http://endpoint",
        model="model-name",
        name="override-name",
        max_concurrency=1,
    )

    assert isinstance(job, DummyBuilderJob)
    assert job.endpoint == "http://endpoint"
    assert job.model == "model-name"
    assert job.name == "payload-name"
    assert job.max_concurrency == 4


def test_to_class_path_and_kwargs_json_roundtrip():
    job = DummyBuilderJob(endpoint="http://endpoint", model="model-name", max_concurrency=3)
    class_path = JobBuilder.to_class_path(job)
    kwargs_json = JobBuilder.to_kwargs_json(job)

    rebuilt = JobBuilder.from_class_path(
        class_path,
        kwargs_json,
        endpoint="http://endpoint",
        model="model-name",
    )
    payload = json.loads(kwargs_json)

    assert class_path == f"{__name__}:DummyBuilderJob"
    assert payload["max_concurrency"] == 3
    assert "endpoint" not in payload
    assert isinstance(rebuilt, DummyBuilderJob)


def test_from_class_path_rejects_invalid_class_path():
    with pytest.raises(ValueError, match="Expected format"):
        JobBuilder.from_class_path(
            "bad-class-path",
            "{}",
            endpoint="http://endpoint",
            model="model-name",
        )


def test_from_class_path_rejects_missing_module():
    with pytest.raises(ValueError, match="could not be imported"):
        JobBuilder.from_class_path(
            "missing.module:Nope",
            "{}",
            endpoint="http://endpoint",
            model="model-name",
        )


def test_from_class_path_rejects_missing_class():
    with pytest.raises(ValueError, match="not found in module"):
        JobBuilder.from_class_path(
            f"{__name__}:MissingJob",
            "{}",
            endpoint="http://endpoint",
            model="model-name",
        )


def test_from_class_path_rejects_non_swarm_job():
    with pytest.raises(ValueError, match="must inherit from SwarmJob"):
        JobBuilder.from_class_path(
            f"{__name__}:NotAJob",
            "{}",
            endpoint="http://endpoint",
            model="model-name",
        )


def test_from_class_path_rejects_invalid_kwargs_json():
    with pytest.raises(ValueError, match="Invalid job kwargs JSON"):
        JobBuilder.from_class_path(
            f"{__name__}:DummyBuilderJob",
            "{bad",
            endpoint="http://endpoint",
            model="model-name",
        )


def test_from_class_path_rejects_non_object_kwargs_json():
    with pytest.raises(ValueError, match="expected a JSON object"):
        JobBuilder.from_class_path(
            f"{__name__}:DummyBuilderJob",
            "[]",
            endpoint="http://endpoint",
            model="model-name",
        )


def test_to_kwargs_json_rejects_non_dict():
    job = InvalidKwargsJob(endpoint="http://endpoint", model="model-name")

    with pytest.raises(ValueError, match="must return a dict"):
        JobBuilder.to_kwargs_json(job)


def test_to_kwargs_json_rejects_unserializable_payload():
    job = UnserializableKwargsJob(endpoint="http://endpoint", model="model-name")

    with pytest.raises(ValueError, match="not JSON-serializable"):
        JobBuilder.to_kwargs_json(job)
