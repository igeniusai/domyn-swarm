from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest

from domyn_swarm.utils.env_path import EnvPath


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("ENDPOINT", "http://localhost")


@pytest.fixture
def mock_launch_reverse_proxy(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "domyn_swarm.helpers.reverse_proxy.get_unused_port", lambda: 54321
    )

    monkeypatch.setattr(
        "domyn_swarm.helpers.reverse_proxy.generate_nginx_config",
        lambda *args, **kwargs: "# fake nginx config",
    )

    called_launch = {}

    def fake_launch_nginx_singularity(sif_path, conf_path, html_path):
        called_launch["called"] = True
        called_launch["sif_path"] = EnvPath(sif_path)
        called_launch["conf_path"] = EnvPath(conf_path)
        called_launch["html_path"] = EnvPath(html_path)

    monkeypatch.setattr(
        "domyn_swarm.helpers.reverse_proxy.launch_nginx_singularity",
        fake_launch_nginx_singularity,
    )
    monkeypatch.setattr(
        "domyn_swarm.helpers.reverse_proxy.run_command", lambda cmd: "fakeuser"
    )
    monkeypatch.setattr(
        "domyn_swarm.helpers.reverse_proxy.get_login_node_suffix", lambda: "42"
    )

    return {
        "called_launch": called_launch,
        "temp_dir": tmp_path,
    }


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
def dummy_swarm_lb_health_checker():
    return Mock(
        jobid=123,
        lb_jobid=456,
        cfg=Mock(lb_port=8080, poll_interval=0.01),
        lb_node=None,
    )
