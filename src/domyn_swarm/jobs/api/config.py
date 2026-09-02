# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""The single statement of what a job's configuration is.

Every field is configuration; `request_params` is the one place provider
request passthrough lives. Keeping the two apart is what lets a provider
parameter share a name with a configuration field without ambiguity.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator


class OutputJoinMode(str, Enum):
    """How a job's outputs are joined back onto its input."""

    APPEND = "append"
    REPLACE = "replace"
    IO_ONLY = "io_only"


class JobConfig(BaseModel):
    """Configuration for a `SwarmJob`.

    Attributes:
        name: Job name used in logs. `None` is resolved by the caller,
            typically to the class name.
        endpoint: LLM endpoint URL. `None` is resolved by the caller,
            typically by falling back to the `ENDPOINT` env var.
        model: Model identifier. Required.
        provider: LLM provider name.
        client_kwargs: Extra kwargs for the async client constructor.
        input_column_name: Column holding the items passed to the job.
        id_column_name: Column holding stable row ids, when the caller supplies one.
        output_cols: Output column name, or names for a multi-column job.
            `None` resolves to `"result"`.
        default_output_cols: Output columns as a list; derived from `output_cols`
            when not given explicitly.
        output_mode: How outputs are joined back onto the input.
        checkpoint_interval: Items processed between checkpoint flushes.
        max_concurrency: Maximum concurrent in-flight requests.
        retries: Retry attempts for a failed request.
        timeout: Request timeout in seconds.
        data_backend: Backend used for IO and conversions.
        native_backend: Whether to use the backend's native execution path.
        backend_read_kwargs: Extra kwargs forwarded to the backend's read.
        backend_write_kwargs: Extra kwargs forwarded to the backend's write.
        native_batch_size: Batch size for the native execution path.
        request_params: Parameters forwarded verbatim to the provider on every
            request. Names here may collide with field names above without
            ambiguity.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    name: str | None = None
    endpoint: str | None = None
    model: str = ""
    provider: str = "openai"
    client_kwargs: dict | None = None

    input_column_name: str = "messages"
    id_column_name: str | None = None
    output_cols: str | list | None = "result"
    default_output_cols: list[str] | None = None
    output_mode: OutputJoinMode = OutputJoinMode.APPEND

    checkpoint_interval: int = 16
    max_concurrency: int = 2
    retries: int = 5
    timeout: float = 600

    data_backend: str | None = None
    native_backend: bool = False
    backend_read_kwargs: dict | None = None
    backend_write_kwargs: dict | None = None
    native_batch_size: int | None = None

    request_params: dict[str, Any] = {}

    @model_validator(mode="after")
    def _resolve_output_cols_and_derive_default(self) -> JobConfig:
        """Resolve `output_cols` and fill `default_output_cols` from it.

        An unspecified or explicit `None` `output_cols` resolves to
        `"result"`. `default_output_cols` becomes a one-element list for a
        string `output_cols` and the list as-is for a list one, and is left
        alone when given explicitly.

        Returns:
            This config, with `output_cols` and `default_output_cols`
            resolved.
        """
        if self.output_cols is None:
            object.__setattr__(self, "output_cols", "result")
        if self.default_output_cols is None:
            derived = [self.output_cols] if isinstance(self.output_cols, str) else self.output_cols
            object.__setattr__(self, "default_output_cols", derived)
        return self

    def merged_with(self, **overrides: Any) -> JobConfig:
        """Return a copy with `overrides` applied.

        Fields this config never received an explicit value for -- including
        `default_output_cols` when it was only derived -- are left for the
        new config to resolve or derive on its own, rather than carried over
        as if they had been explicit.

        Args:
            **overrides: Field values taking precedence over this config's.

        Returns:
            A new, validated config.
        """
        data = self.model_dump(exclude_unset=True)
        data.update(overrides)
        return type(self)(**data)
