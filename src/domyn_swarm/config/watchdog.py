from typing import Literal

from pydantic import BaseModel, Field, field_validator


class WatchdogRayConfig(BaseModel):
    enabled: bool = False
    expected_tp: int | None = None
    probe_timeout_s: float = 120.0
    status_grace_s: float = 10.0
    probe_interval_s: float = 10.0


class WatchdogConfig(BaseModel):
    enabled: bool = True
    probe_interval: int = 5
    http_path: str = "/health"
    http_timeout: float = 2.0
    readiness_timeout: int = 600

    restart_policy: Literal["always", "on-failure", "never"] = "on-failure"
    unhealthy_restart_after: int = 120
    max_restarts: int = 3
    restart_backoff_initial: int = 5
    restart_backoff_max: int = 60

    kill_grace_seconds: int = 10
    log_level: Literal["debug", "info", "warning", "error"] = "info"

    ray: WatchdogRayConfig = Field(default_factory=WatchdogRayConfig)

    @field_validator("http_path")
    @classmethod
    def _ensure_leading_slash(cls, v: str) -> str:
        return v if v.startswith("/") else f"/{v}"
