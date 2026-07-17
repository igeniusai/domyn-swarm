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
