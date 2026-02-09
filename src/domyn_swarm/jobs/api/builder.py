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

"""Helpers to serialize and reconstruct ``SwarmJob`` instances."""

import importlib
import json
from typing import Any, cast

from .base import SwarmJob


class JobBuilder:
    """Factory and serializer for ``SwarmJob`` instances."""

    @staticmethod
    def from_class_path(job_class: str, kwargs_json: str, **overrides: Any) -> SwarmJob:
        """Build a ``SwarmJob`` from class path and JSON kwargs.

        Args:
            job_class: Job class in ``module:ClassName`` format.
            kwargs_json: JSON object containing constructor kwargs.
            **overrides: Constructor kwargs applied before payload kwargs.

        Returns:
            Instantiated ``SwarmJob``.

        Raises:
            ValueError: If class path, payload, or instantiation is invalid.
        """
        module_name, class_name = JobBuilder._parse_class_path(job_class)
        job_type = JobBuilder._load_job_type(module_name, class_name, class_path=job_class)
        payload = JobBuilder._parse_kwargs_payload(kwargs_json)
        init_kwargs = {**overrides, **payload}

        try:
            job = job_type(**init_kwargs)
        except Exception as exc:
            raise ValueError(f"Failed to instantiate job '{job_class}': {exc}") from exc

        return job

    @staticmethod
    def to_class_path(job: SwarmJob) -> str:
        """Serialize a job instance to ``module:ClassName`` format.

        Args:
            job: Job instance.

        Returns:
            Fully-qualified class path.
        """
        return f"{job.__class__.__module__}:{job.__class__.__qualname__}"

    @staticmethod
    def to_kwargs_json(job: SwarmJob) -> str:
        """Serialize job constructor kwargs as JSON.

        Args:
            job: Job instance.

        Returns:
            JSON string for constructor kwargs.

        Raises:
            ValueError: If kwargs extraction or serialization fails.
        """
        try:
            payload = job.to_kwargs()
        except Exception as exc:
            raise ValueError(
                f"Failed to extract kwargs from job '{job.__class__.__name__}': {exc}"
            ) from exc

        if not isinstance(payload, dict):
            payload_type = type(payload).__name__
            raise ValueError(
                f"Job '{job.__class__.__name__}' to_kwargs() must return a dict, got "
                f"{payload_type}."
            )

        try:
            return json.dumps(payload)
        except TypeError as exc:
            raise ValueError(
                f"Job '{job.__class__.__name__}' kwargs are not JSON-serializable: {exc}"
            ) from exc

    @staticmethod
    def _parse_class_path(job_class: str) -> tuple[str, str]:
        """Parse ``module:ClassName`` specification.

        Args:
            job_class: Job class specification.

        Returns:
            ``(module_name, class_name)`` tuple.

        Raises:
            ValueError: If class path format is invalid.
        """
        if ":" not in job_class:
            raise ValueError(
                f"Invalid job class path '{job_class}'. Expected format 'module:ClassName'."
            )

        module_name, class_name = job_class.split(":", 1)
        if not module_name or not class_name:
            raise ValueError(
                f"Invalid job class path '{job_class}'. Expected format 'module:ClassName'."
            )
        return module_name, class_name

    @staticmethod
    def _load_job_type(module_name: str, class_name: str, *, class_path: str) -> type[SwarmJob]:
        """Resolve and validate a ``SwarmJob`` class.

        Args:
            module_name: Python module path.
            class_name: Class name in module.
            class_path: Original class path string.

        Returns:
            Concrete ``SwarmJob`` type.

        Raises:
            ValueError: If module/class cannot be resolved or is not a ``SwarmJob``.
        """
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            raise ValueError(f"Job module '{module_name}' could not be imported.") from exc

        if not hasattr(module, class_name):
            raise ValueError(f"Job class '{class_name}' not found in module '{module_name}'.")

        job_type = getattr(module, class_name)
        if not isinstance(job_type, type):
            raise ValueError(f"Job target '{class_path}' is not a class.")
        if not issubclass(job_type, SwarmJob):
            raise ValueError(f"Job class '{class_path}' must inherit from SwarmJob.")

        return cast(type[SwarmJob], job_type)

    @staticmethod
    def _parse_kwargs_payload(kwargs_json: str) -> dict[str, Any]:
        """Parse and validate kwargs JSON payload.

        Args:
            kwargs_json: JSON object string.

        Returns:
            Decoded payload dictionary.

        Raises:
            ValueError: If payload is not a valid JSON object.
        """
        try:
            payload = json.loads(kwargs_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid job kwargs JSON: {exc.msg}.") from exc

        if not isinstance(payload, dict):
            raise ValueError("Invalid job kwargs JSON: expected a JSON object.")
        return payload
