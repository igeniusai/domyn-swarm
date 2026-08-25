# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class WatchdogRayConfig(BaseModel):
    enabled: bool = Field(
        default=False,
        description=(
            "Enable Ray-aware health checks (cluster liveness and capacity) in "
            "addition to the HTTP checks."
        ),
    )
    expected_tp: int | None = Field(
        default=None,
        description=(
            "Expected tensor-parallel world size, i.e. the total GPU count for "
            "vLLM. Used by the Ray capacity check; `None` disables capacity "
            "enforcement."
        ),
    )
    probe_timeout_s: float = Field(
        default=120.0,
        description=(
            "Seconds allowed for each Ray health probe command, such as "
            "`ray status` or `ray list nodes`."
        ),
    )
    status_grace_s: float = Field(
        default=10.0,
        description=(
            "Ray must report healthy for at least this many seconds before the "
            "watchdog treats it as fully ready."
        ),
    )
    probe_interval_s: float = Field(
        default=30.0,
        description="Seconds between Ray health probes when Ray checks are enabled.",
    )


class WatchdogConfig(BaseModel):
    enabled: bool = Field(
        default=True,
        description="Master switch for the per-replica watchdog process.",
    )
    probe_interval: int = Field(
        default=30,
        description="Seconds between watchdog HTTP and Ray health probes.",
    )
    http_path: str = Field(
        default="/health",
        description=(
            "HTTP path probed on the vLLM REST server to determine readiness and "
            "health. A leading `/` is added automatically if missing."
        ),
    )
    http_timeout: float = Field(
        default=2.0,
        description="Seconds allowed for each HTTP health probe request.",
    )
    readiness_timeout: int = Field(
        default=600,
        description=(
            "Seconds the server is given to become ready before it is considered unhealthy."
        ),
    )

    restart_policy: Literal["always", "on-failure", "never"] = Field(
        default="on-failure",
        description="When the watchdog should restart the child vLLM process.",
    )
    unhealthy_restart_after: int = Field(
        default=120,
        description=(
            "If the replica stays unhealthy for this many seconds, the watchdog "
            "forces a restart, or exits, depending on the restart policy."
        ),
    )
    max_restarts: int = Field(
        default=3,
        description=(
            "Maximum restart attempts before giving up and leaving the replica in the failed state."
        ),
    )
    restart_backoff_initial: int = Field(
        default=5,
        description="Seconds to wait before the first restart attempt.",
    )
    restart_backoff_max: int = Field(
        default=60,
        description=(
            "Upper bound in seconds for the exponential backoff between restart attempts."
        ),
    )

    kill_grace_seconds: int = Field(
        default=10,
        description=(
            "Seconds to wait after SIGTERM before the watchdog sends SIGKILL to the child process."
        ),
    )
    log_level: Literal["debug", "info", "warning", "error"] = Field(
        default="info",
        description="Log verbosity for the watchdog process.",
    )

    ray: WatchdogRayConfig = Field(
        default_factory=WatchdogRayConfig,
        description=(
            "Ray-aware health checking, layered on top of the HTTP probe. Disabled "
            "unless `ray.enabled` is set."
        ),
    )

    @field_validator("http_path")
    @classmethod
    def _ensure_leading_slash(cls, v: str) -> str:
        return v if v.startswith("/") else f"/{v}"
