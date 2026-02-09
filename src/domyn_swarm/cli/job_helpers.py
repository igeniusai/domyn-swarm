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

"""Helper utilities for ``domyn-swarm job`` CLI commands."""

import contextlib
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, RootModel, ValidationError, field_validator
import typer

from domyn_swarm.core.state.state_manager import SwarmStateManager
from domyn_swarm.core.swarm import DomynLLMSwarm
from domyn_swarm.jobs.api import JobBuilder
from domyn_swarm.jobs.api.base import SwarmJob
from domyn_swarm.platform.protocols import JobHandle, JobStatus, coerce_job_status


class JsonObjectModel(RootModel[dict[str, Any]]):
    """Pydantic wrapper for validating JSON object payloads."""


class HandlePayloadModel(BaseModel):
    """Validated handle payload from ``--handle-json`` or stdin.

    Attributes:
        command: Optional source command.
        swarm: Optional swarm deployment name.
        id: Optional provider/job handle ID.
        job_id: Optional internal job ID.
        status: Optional normalized status.
        pid: Optional process ID.
        external_id: Optional provider external ID.
    """

    model_config = ConfigDict(extra="allow")

    command: str | None = None
    swarm: str | None = None
    id: str | None = None
    job_id: str | None = None
    status: JobStatus | None = None
    pid: int | None = None
    external_id: str | None = None

    @field_validator("status", mode="before")
    @classmethod
    def _normalize_status(cls, status: object) -> JobStatus | None:
        """Normalize incoming status payload.

        Args:
            status: Raw status value.

        Returns:
            Normalized status or ``None`` if missing.
        """
        if status is None:
            return None
        return coerce_job_status(status)


@dataclass(frozen=True)
class JobRunSpec:
    """Run-time parameters for job submission."""

    input_path: Path
    output_path: Path
    shard_output: bool
    checkpoint_dir: Path | None
    no_resume: bool
    no_checkpointing: bool
    runner: str
    num_threads: int
    limit: int | None
    detach: bool
    mail_user: str | None
    ray_address: str | None
    global_resume: bool = False
    checkpoint_tag: str | None = None
    shard_mode: Literal["id", "index"] = "id"


@dataclass(frozen=True)
class JobSubmitRequest:
    """Job submission payload for helper calls."""

    job: SwarmJob
    run: JobRunSpec


@dataclass(frozen=True)
class ResolvedJobTarget:
    """Resolved target used by job wait/cancel commands."""

    swarm_name: str
    handle: JobHandle
    job_id: str | None
    source: Literal["job_id", "external_id", "handle_json", "stdin"]


def parse_json_object(value: str, *, param_name: str) -> dict[str, Any]:
    """Parse and validate a JSON object payload.

    Args:
        value: JSON string to parse.
        param_name: Parameter name used for error messages.

    Returns:
        Parsed JSON object.

    Raises:
        typer.BadParameter: If payload is invalid or not a JSON object.
    """
    try:
        return JsonObjectModel.model_validate_json(value).root
    except ValidationError as exc:
        raise typer.BadParameter(f"{param_name} must be a valid JSON object.") from exc


def maybe_parse_json_object(value: str | None, *, param_name: str) -> dict[str, Any] | None:
    """Parse optional JSON object payload.

    Args:
        value: Optional JSON payload.
        param_name: Parameter name used for error messages.

    Returns:
        Parsed dictionary or ``None``.
    """
    if value is None:
        return None
    return parse_json_object(value, param_name=param_name)


def build_job_kwargs_json(
    *,
    job_kwargs: str,
    data_backend: str | None,
    native_backend: bool | None,
    native_batch_size: int | None,
    id_column_name: str | None,
    backend_read_kwargs: str | None,
    backend_write_kwargs: str | None,
) -> str:
    """Build normalized constructor kwargs payload for job instantiation.

    Args:
        job_kwargs: JSON job kwargs payload.
        data_backend: Optional data backend override.
        native_backend: Optional native backend flag.
        native_batch_size: Optional native batch size.
        id_column_name: Optional stable ID column name.
        backend_read_kwargs: Optional backend read kwargs JSON payload.
        backend_write_kwargs: Optional backend write kwargs JSON payload.

    Returns:
        Serialized JSON payload for job constructor.
    """
    payload = parse_json_object(job_kwargs, param_name="job_kwargs")
    if data_backend:
        payload["data_backend"] = data_backend
    if native_backend is not None:
        payload["native_backend"] = native_backend
    if native_batch_size is not None:
        payload["native_batch_size"] = native_batch_size
    if id_column_name is not None:
        payload["id_column_name"] = id_column_name

    read_kwargs = maybe_parse_json_object(backend_read_kwargs, param_name="backend_read_kwargs")
    if read_kwargs is not None:
        payload["backend_read_kwargs"] = read_kwargs

    write_kwargs = maybe_parse_json_object(backend_write_kwargs, param_name="backend_write_kwargs")
    if write_kwargs is not None:
        payload["backend_write_kwargs"] = write_kwargs

    resolved_backend = payload.get("data_backend")
    if resolved_backend == "ray" and not payload.get("id_column_name"):
        raise typer.BadParameter("--id-column is required when --data-backend=ray.")

    return json.dumps(payload)


def build_job_for_swarm(
    *,
    swarm: DomynLLMSwarm,
    job_class: str,
    job_kwargs: str,
    job_name: str | None,
    input_column: str,
    output_column: str,
    checkpoint_interval: int,
    max_concurrency: int,
    retries: int,
    timeout: float,
) -> SwarmJob:
    """Build a ``SwarmJob`` instance for a specific swarm context.

    Args:
        swarm: Active swarm instance.
        job_class: Job class path.
        job_kwargs: Serialized constructor kwargs.
        job_name: Optional job name override.
        input_column: Input column name.
        output_column: Output column name.
        checkpoint_interval: Checkpoint interval.
        max_concurrency: Max concurrency.
        retries: Retry count.
        timeout: Request timeout.

    Returns:
        Instantiated job object.
    """
    return JobBuilder.from_class_path(
        job_class,
        job_kwargs,
        name=job_name,
        endpoint=swarm.endpoint,
        model=swarm.model,
        checkpoint_interval=checkpoint_interval,
        max_concurrency=max_concurrency,
        retries=retries,
        timeout=timeout,
        input_column_name=input_column,
        output_cols=output_column,
    )


def normalize_submission_handle(raw_handle: object) -> JobHandle:
    """Normalize arbitrary backend handle values into ``JobHandle``.

    Args:
        raw_handle: Backend returned handle object.

    Returns:
        Normalized ``JobHandle``.
    """
    if isinstance(raw_handle, JobHandle):
        return raw_handle

    raw_meta = getattr(raw_handle, "meta", None)
    return JobHandle(
        id=str(getattr(raw_handle, "id", "unknown")),
        status=coerce_job_status(getattr(raw_handle, "status", JobStatus.PENDING)),
        meta=dict(raw_meta) if isinstance(raw_meta, dict) else {},
    )


def emit_submission_json(
    *,
    handle: object,
    command: Literal["submit", "submit-script"],
    swarm_name: str,
) -> None:
    """Emit parseable JSON payload for submission commands.

    Args:
        handle: Submission handle.
        command: Command name.
        swarm_name: Swarm deployment name.
    """
    normalized = normalize_submission_handle(handle)
    payload = {
        "command": command,
        "swarm": str(swarm_name),
        "id": normalized.id,
        "job_id": normalized.meta.get("job_id"),
        "status": normalized.status.value,
        "pid": normalized.pid,
        "external_id": normalized.external_id,
        "detach": normalized.pid is not None,
    }
    typer.echo(json.dumps(payload, separators=(",", ":"), sort_keys=True))


def submit_loaded_job(*, swarm: DomynLLMSwarm, request: JobSubmitRequest) -> JobHandle:
    """Submit an already-instantiated job.

    Args:
        swarm: Target swarm.
        request: Submission request payload.

    Returns:
        Submitted job handle.
    """
    resolved_checkpoint_dir = (
        swarm.swarm_dir / "checkpoints"
        if request.run.checkpoint_dir is None
        else request.run.checkpoint_dir
    )
    return swarm.submit_job(
        request.job,
        input_path=request.run.input_path,
        output_path=request.run.output_path,
        num_threads=request.run.num_threads,
        shard_output=request.run.shard_output,
        limit=request.run.limit,
        detach=request.run.detach,
        mail_user=request.run.mail_user,
        checkpoint_dir=resolved_checkpoint_dir,
        no_resume=request.run.no_resume,
        no_checkpointing=request.run.no_checkpointing,
        runner=request.run.runner,
        ray_address=request.run.ray_address,
        checkpoint_tag=request.run.checkpoint_tag,
        shard_mode=request.run.shard_mode,
        global_resume=request.run.global_resume,
    )


def maybe_cancel_swarm_on_keyboard_interrupt(swarm_ctx: DomynLLMSwarm) -> None:
    """Prompt whether to cancel swarm allocation on keyboard interrupt.

    Args:
        swarm_ctx: Swarm context to cleanup on user confirmation.
    """
    abort = typer.confirm("KeyboardInterrupt detected. Do you want to cancel the swarm allocation?")
    if abort:
        with contextlib.suppress(Exception):
            swarm_ctx.cleanup()
        typer.echo("Swarm allocation cancelled by user")
        raise typer.Abort() from None
    typer.echo("Continuing to wait for job to complete …")


def _coerce_optional_int(value: object) -> int | None:
    """Coerce optional PID values from payloads."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _job_record_to_handle(record: dict[str, Any], *, pid: int | None = None) -> JobHandle:
    """Build a ``JobHandle`` from a persisted job record."""
    external_id = record.get("external_id")
    job_id = record.get("job_id")
    handle_id = str(external_id or job_id or "unknown")
    meta: dict[str, Any] = {
        "job_id": job_id,
        "external_id": external_id,
    }
    if pid is not None:
        meta["pid"] = pid
    return JobHandle(
        id=handle_id,
        status=coerce_job_status(record.get("status", JobStatus.PENDING)),
        meta=meta,
    )


def _parse_handle_payload(raw: str, *, source: str) -> HandlePayloadModel:
    """Parse and validate handle payload.

    Args:
        raw: Raw JSON payload.
        source: Source label for error messaging.

    Returns:
        Validated payload model.

    Raises:
        typer.BadParameter: If payload is invalid.
    """
    try:
        return HandlePayloadModel.model_validate_json(raw)
    except ValidationError as exc:
        raise typer.BadParameter(f"{source} must be a valid JSON object.") from exc


def _read_handle_payload(
    handle_json: str | None,
) -> tuple[HandlePayloadModel | None, Literal["handle_json", "stdin"] | None]:
    """Read handle payload from option value or stdin.

    Args:
        handle_json: ``--handle-json`` option value. Use ``-`` for stdin.

    Returns:
        Tuple containing optional payload and source.
    """
    if handle_json is not None:
        if handle_json == "-":
            raw = sys.stdin.read().strip()
            if not raw:
                raise typer.BadParameter("Expected JSON payload on stdin for --handle-json=-.")
            return _parse_handle_payload(raw, source="handle_json"), "stdin"
        return _parse_handle_payload(handle_json, source="handle_json"), "handle_json"

    if sys.stdin.isatty():
        return None, None

    raw = sys.stdin.read().strip()
    if not raw:
        return None, None
    return _parse_handle_payload(raw, source="stdin"), "stdin"


def resolve_job_target(
    *,
    job_id: str | None,
    external_id: str | None,
    handle_json: str | None,
    deployment_name: str | None,
) -> ResolvedJobTarget:
    """Resolve CLI selectors to a concrete target for wait/cancel.

    Args:
        job_id: Internal job ID selector.
        external_id: Provider external-id selector.
        handle_json: Handle payload selector.
        deployment_name: Optional deployment-name hint.

    Returns:
        Resolved target bundle.

    Raises:
        typer.BadParameter: If selectors are invalid or cannot be resolved.
    """
    payload, payload_source = _read_handle_payload(handle_json)
    selectors = int(job_id is not None) + int(external_id is not None) + int(payload is not None)
    if selectors != 1:
        raise typer.BadParameter(
            "Provide exactly one selector: --job-id, --external-id, or --handle-json (or stdin)."
        )

    try:
        if job_id is not None:
            record = SwarmStateManager.get_job(job_id)
            return ResolvedJobTarget(
                swarm_name=str(record["deployment_name"]),
                handle=_job_record_to_handle(record),
                job_id=str(record["job_id"]),
                source="job_id",
            )

        if external_id is not None:
            record = SwarmStateManager.get_job_by_external_id(
                external_id,
                deployment_name=deployment_name,
            )
            return ResolvedJobTarget(
                swarm_name=str(record["deployment_name"]),
                handle=_job_record_to_handle(record),
                job_id=str(record["job_id"]),
                source="external_id",
            )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    assert payload is not None
    source = "stdin" if payload_source == "stdin" else "handle_json"
    payload_swarm = payload.swarm or deployment_name
    payload_pid = _coerce_optional_int(payload.pid)

    try:
        if payload.job_id:
            record = SwarmStateManager.get_job(payload.job_id)
            return ResolvedJobTarget(
                swarm_name=str(record["deployment_name"]),
                handle=_job_record_to_handle(record, pid=payload_pid),
                job_id=str(record["job_id"]),
                source=source,
            )

        if payload.external_id:
            record = SwarmStateManager.get_job_by_external_id(
                payload.external_id,
                deployment_name=str(payload_swarm) if payload_swarm else None,
            )
            return ResolvedJobTarget(
                swarm_name=str(record["deployment_name"]),
                handle=_job_record_to_handle(record, pid=payload_pid),
                job_id=str(record["job_id"]),
                source=source,
            )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if not payload_swarm:
        raise typer.BadParameter(
            "Handle JSON without job_id/external_id must include 'swarm' (or pass --name)."
        )

    handle = JobHandle(
        id=str(payload.id or payload.external_id or "unknown"),
        status=payload.status or JobStatus.PENDING,
        meta={
            "job_id": None,
            "external_id": payload.external_id,
            "pid": payload_pid,
        },
    )
    return ResolvedJobTarget(
        swarm_name=str(payload_swarm),
        handle=handle,
        job_id=None,
        source=source,
    )


def emit_job_control_json(
    *,
    command: Literal["wait", "cancel"],
    target: ResolvedJobTarget,
    status: JobStatus,
) -> None:
    """Emit parseable JSON payload for job wait/cancel commands.

    Args:
        command: Command name.
        target: Resolved target.
        status: Final status.
    """
    payload = {
        "command": command,
        "resolved_by": target.source,
        "swarm": target.swarm_name,
        "id": target.handle.id,
        "job_id": target.job_id or target.handle.meta.get("job_id"),
        "status": status.value,
        "pid": target.handle.pid,
        "external_id": target.handle.external_id,
    }
    typer.echo(json.dumps(payload, separators=(",", ":"), sort_keys=True))
