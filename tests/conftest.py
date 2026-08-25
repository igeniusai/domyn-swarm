# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock

import pandas as pd
import pytest

from domyn_swarm.utils.env_path import EnvPath


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("ENDPOINT", "http://localhost")


@pytest.fixture
def parquet_file(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    file_path = tmp_path / "test.parquet"
    df.to_parquet(file_path)
    return EnvPath(file_path)


@pytest.fixture
def mock_client(monkeypatch):
    mock = AsyncMock()
    monkeypatch.setattr("openai.OpenAI", lambda *args, **kwargs: mock)
    return mock


@pytest.fixture
def disable_autoupgrade(monkeypatch):
    monkeypatch.setenv("DOMYN_SWARM_SKIP_DB_UPGRADE", "1")


@pytest.fixture(autouse=True)
def clear_settings_cache_between_tests():
    from domyn_swarm.config.settings import reload_settings_cache

    # before each test
    reload_settings_cache()
    yield
    # after each test (optional)
    reload_settings_cache()


@pytest.fixture(autouse=True)
def isolate_defaults_file(monkeypatch, tmp_path):
    """Hide any developer-local ``defaults.yaml`` from the test run.

    ``config.defaults`` resolves overridable defaults from ``$DOMYN_SWARM_DEFAULTS``,
    the cwd and the home directory. CI has none of those files, so a config that
    silently leans on a *required* default (e.g. ``slurm.endpoint.nginx_image``)
    passes on a machine that happens to have one and fails in CI. Pin the search
    to "nothing found" so local runs match CI; tests that exercise the loader
    itself mock ``get_settings``/``_DEFAULT_FILES`` and override this.
    """
    from domyn_swarm.config import defaults as defaults_mod
    from domyn_swarm.config.settings import reload_settings_cache

    monkeypatch.setenv("DOMYN_SWARM_DEFAULTS", str(tmp_path / "no-such-defaults.yaml"))
    monkeypatch.setattr(defaults_mod, "_DEFAULT_FILES", ())
    reload_settings_cache()
    defaults_mod.reload_defaults_cache()
    yield
    defaults_mod.reload_defaults_cache()
