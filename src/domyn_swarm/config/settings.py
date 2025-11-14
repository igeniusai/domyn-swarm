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

from functools import lru_cache
from pathlib import Path

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
    home: Path = Field(
        default=Path("~/.domyn_swarm").expanduser(),
        description="Path to domyn-swarm home directory",
    )
    # Path to YAML with overridable defaults (used by your defaults loader)
    defaults_file: Path | None = Field(
        default_factory=lambda data: data["home"] / "defaults.yaml",
        alias="DOMYN_SWARM_DEFAULTS",
        description="Path to YAML with overridable defaults (used by your defaults loader)",
    )

    # --- Secrets / tokens ----------------------------------------------------
    api_token: SecretStr | None = Field(
        default=None,
        description="API token for authenticating with the domyn-swarm vllm server",
    )  # DOMYN_SWARM_API_TOKEN
    vllm_api_key: SecretStr | None = Field(
        default_factory=lambda data: data.get("api_token"),
        alias="VLLM_API_KEY",
        description="Alternative env var for API token, used by vLLM",
    )
    singularityenv_vllm_api_key: SecretStr | None = Field(
        default_factory=lambda data: data.get("vllm_api_key"),
        alias="SINGULARITYENV_VLLM_API_KEY",
        description="Alternative env var for API token, used inside Singularity containers",
    )

    # --- Slurm ---------------------------------------------------------------
    mail_user: str | None = Field(
        description="Email address for Slurm job notifications (if enabled)",
        default=None,
    )  # DOMYN_SWARM_MAIL_USER

    # --- Lepton --------------------------------------------------------------
    lepton_api_token: SecretStr | None = Field(
        default=None,
        alias="LEPTONAI_API_TOKEN",
        description="API token for authenticating with Lepton AI",
    )
    lepton_workspace_id: str | None = Field(
        default=None,
        alias="LEPTON_WORKSPACE_ID",
        description="Workspace ID for Lepton AI",
    )

    # --- CLI / TUI -----------------------------------------------------------
    ascii: bool = Field(
        default=False,
        description="Use ASCII glyphs instead of Unicode emojis in CLI output",
    )  # DOMYN_SWARM_ASCII

    skip_db_upgrade: bool = Field(
        default=False,
        description="If true, skip automatic database schema upgrades on CLI startup",
    )  # DOMYN_SWARM_SKIP_DB_UPGRADE

    # --- AzureML (placeholders) ---------------------------------------------
    # azure_subscription_id: Optional[str] = None
    # azure_resource_group: Optional[str] = None
    # azure_workspace_name: Optional[str] = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Cached accessor. Call this wherever you need settings.
    Tests can `cache_clear()` before reading to pick up monkeypatched env.
    """
    return Settings()


def reload_settings_cache() -> None:
    get_settings.cache_clear()  # type: ignore[attr-defined]
