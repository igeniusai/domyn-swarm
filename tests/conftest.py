import pandas as pd
import pytest

from domyn_swarm.utils.env_path import EnvPath

@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("ENDPOINT", "http://localhost")


@pytest.fixture
def mock_launch_reverse_proxy(monkeypatch, tmp_path):
    monkeypatch.setattr("domyn_swarm.helpers.get_unused_port", lambda: 54321)

    monkeypatch.setattr(
        "domyn_swarm.helpers.generate_nginx_config",
        lambda *args, **kwargs: "# fake nginx config",
    )

    called_launch = {}

    def fake_launch_nginx_singularity(sif_path, conf_path, html_path):
        called_launch["called"] = True
        called_launch["sif_path"] = EnvPath(sif_path)
        called_launch["conf_path"] = EnvPath(conf_path)
        called_launch["html_path"] = EnvPath(html_path)

    monkeypatch.setattr(
        "domyn_swarm.helpers.launch_nginx_singularity", fake_launch_nginx_singularity
    )
    monkeypatch.setattr("domyn_swarm.helpers.run_command", lambda cmd: "fakeuser")
    monkeypatch.setattr("domyn_swarm.helpers.get_login_node_suffix", lambda: "42")

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
