from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized environment configuration for domyn-swarm.

    Env var naming: DOMYN_SWARM_<FIELD_NAME> (custom aliases below).
    A .env file in CWD or ~/.domyn_swarm/.env is read automatically.
    """

    model_config = SettingsConfigDict(
        env_prefix="DOMYN_SWARM_",
        env_file=(".env", "~/.domyn_swarm/.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- General -------------------------------------------------------------
    log_level: str = "INFO"
    home: Path = Field(default=Path("~/.domyn_swarm").expanduser())
    # Path to YAML with overridable defaults (used by your defaults loader)
    defaults_file: Optional[Path] = Field(
        default_factory=lambda data: data["home"] / "defaults.yaml",
        alias="DOMYN_SWARM_DEFAULTS",
    )

    # --- Secrets / tokens ----------------------------------------------------
    api_token: Optional[SecretStr] = Field(
        default=None,
        alias="API_TOKEN",
        description="API token for authenticating with the domyn-swarm vllm server",
    )

    # --- Slurm ---------------------------------------------------------------
    mail_user: Optional[str] = None  # DOMYN_SWARM_MAIL_USER

    # --- Lepton --------------------------------------------------------------
    lepton_api_token: Optional[SecretStr] = Field(
        default=None, alias="LEPTONAI_API_TOKEN"
    )
    lepton_workspace_id: Optional[str] = Field(
        default=None, alias="LEPTON_WORKSPACE_ID"
    )

    # --- AzureML (placeholders) ---------------------------------------------
    azure_subscription_id: Optional[str] = None
    azure_resource_group: Optional[str] = None
    azure_workspace_name: Optional[str] = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Cached accessor. Call this wherever you need settings.
    Tests can `cache_clear()` before reading to pick up monkeypatched env.
    """
    return Settings()


def reload_settings_cache() -> None:
    get_settings.cache_clear()  # type: ignore[attr-defined]
