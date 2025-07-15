import math
import socket
import subprocess
import time

import pandas as pd
import pytest
from domyn_swarm.helpers import compute_perplexity, compute_perplexity_metrics, generate_ssh_tunnel_cmd, get_login_node_suffix, get_unused_port, launch_nginx_singularity, launch_reverse_proxy, parquet_hash, run_command
from domyn_swarm.utils.env_path import EnvPath


def test_get_unused_port_returns_valid_port():
    port = get_unused_port(start=50000, end=51000)
    assert isinstance(port, int)
    assert 50000 <= port <= 51000

    # Try binding to it — should succeed since it’s truly unused
    s = socket.socket()
    try:
        s.bind(("", port))
    finally:
        s.close()


def test_get_unused_port_exhausted_range(monkeypatch):
    # Monkeypatch socket to simulate no free ports
    def always_fail_bind(*args, **kwargs):
        raise OSError("Port in use")

    monkeypatch.setattr(socket, "socket", lambda: type("MockSocket", (), {
        "bind": always_fail_bind,
        "listen": lambda self, n=1: None,
        "close": lambda self: None,
    })())

    with pytest.raises(IOError, match="No free ports available"):
        get_unused_port(start=60000, end=60001)


def test_get_login_node_suffix_success(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *args, **kwargs: "lrdn1234.domain.local\n"
    )

    result = get_login_node_suffix()
    assert result == "34"


def test_get_login_node_suffix_error(monkeypatch, capsys):
    def raise_error(*args, **kwargs):
        raise subprocess.CalledProcessError(1, cmd="hostname")

    monkeypatch.setattr(subprocess, "check_output", raise_error)

    with pytest.raises(SystemExit) as exc:
        get_login_node_suffix()

    assert exc.value.code == 1

    captured = capsys.readouterr()
    assert "Error getting hostname" in captured.err

def test_run_command_success():
    output = run_command("echo 'hello world'")
    assert output == "hello world"


def test_run_command_failure():
    # This will return a non-zero code and stderr
    with pytest.raises(AssertionError, match="Command failed with error"):
        run_command("ls non_existent_file")

def test_generate_ssh_tunnel_cmd_basic():
    cmd = generate_ssh_tunnel_cmd(
        user="fdambro1",
        localhost_port=8888,
        nginx_port=9000,
        login_node_suffix="35"
    )

    expected = (
        "ssh -N -L 8888:login35.leonardo.local:9000 "
        "fdambro1@login35-ext.leonardo.cineca.it"
    )

    assert cmd == expected

def test_launch_nginx_singularity(monkeypatch, tmp_path):
    # Setup fake paths
    sif_path = EnvPath(tmp_path / "nginx.sif")
    conf_path = EnvPath(tmp_path / "nginx.conf")
    html_path = EnvPath(tmp_path / "html")

    for p in [sif_path, conf_path, html_path]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# dummy")

    # Patch environment and methods
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(time, "sleep", lambda x: None)

    launched_cmds = []

    class FakePopen:
        def __init__(self, cmd, stdout=None, stderr=None):
            launched_cmds.append(cmd)
            self.cmd = cmd
            self.stdout = stdout
            self.stderr = stderr

    monkeypatch.setattr(subprocess, "Popen", FakePopen)

    launch_nginx_singularity(sif_path, conf_path, html_path)

    assert len(launched_cmds) == 2
    stop_cmd, start_cmd = launched_cmds

    assert stop_cmd[:3] == ["singularity", "instance", "stop"]
    assert "nginx_instance" in stop_cmd

    assert start_cmd[:3] == ["singularity", "instance", "start"]
    assert any("nginx.conf" in str(arg) for arg in start_cmd)
    assert any("html" in str(arg) for arg in start_cmd)
    assert any("cache" in str(arg) for arg in start_cmd)
    assert str(sif_path) in start_cmd

    log_path = tmp_path / "nginx_singularity.log"
    assert log_path.exists()

def test_launch_reverse_proxy(mock_launch_reverse_proxy, tmp_path, capsys):
    nginx_template = EnvPath(tmp_path / "nginx_template.j2")
    nginx_template.write_text("# template")

    image_path = EnvPath(tmp_path / "nginx_image.sif")
    image_path.write_text("# sif")

    launch_reverse_proxy(
        nginx_template=nginx_template,
        image_path=image_path,
        lb_node="lrdn001",
        head_node="lrdn001",
        vllm_port=8000,
        ray_dashboard_port=8265,
    )

    out = capsys.readouterr().out
    assert "[INFO] Launching reverse proxy on port 54321" in out
    assert "ssh -N -L 54321:login42.leonardo.local:54321 \nfakeuser@login42-ext.leonardo.cineca.it" in out

    called = mock_launch_reverse_proxy["called_launch"]
    assert called.get("called") is True
    assert "nginx.conf" in called["conf_path"].name
    assert "index.html" in [f.name for f in called["html_path"].rglob("*")]

def test_parquet_hash_blake2b(parquet_file):
    digest = parquet_hash(parquet_file, algorithm="blake2b")
    assert isinstance(digest, str)
    assert len(digest) == 8
    assert all(c in "0123456789abcdef" for c in digest)


def test_parquet_hash_sha256(parquet_file):
    digest = parquet_hash(parquet_file, algorithm="sha256")
    assert isinstance(digest, str)
    assert len(digest) == 8


def test_parquet_hash_md5(parquet_file):
    digest = parquet_hash(parquet_file, algorithm="md5")
    assert isinstance(digest, str)
    assert len(digest) == 8


def test_parquet_hash_invalid_algorithm(parquet_file):
    with pytest.raises(ValueError):
        parquet_hash(parquet_file, algorithm="notarealhash")


def test_parquet_hash_identical_files(tmp_path):
    df = pd.DataFrame({"x": [10, 20], "y": [0.1, 0.2]})
    f1 = tmp_path / "file1.parquet"
    f2 = tmp_path / "file2.parquet"
    df.to_parquet(f1)
    df.to_parquet(f2)

    hash1 = parquet_hash(f1)
    hash2 = parquet_hash(f2)
    assert hash1 == hash2


def test_parquet_hash_different_files(tmp_path):
    df1 = pd.DataFrame({"x": [1]})
    df2 = pd.DataFrame({"x": [2]})
    f1 = tmp_path / "a.parquet"
    f2 = tmp_path / "b.parquet"
    df1.to_parquet(f1)
    df2.to_parquet(f2)

    hash1 = parquet_hash(f1)
    hash2 = parquet_hash(f2)
    assert hash1 != hash2

def test_perplexity_typical_case():
    logprobs = [-1.0, -2.0, -1.5]
    result = compute_perplexity(logprobs)
    expected = math.exp(-sum(logprobs) / len(logprobs))
    assert math.isclose(result, expected)


def test_perplexity_empty_list_returns_inf():
    result = compute_perplexity([])
    assert result == float("inf")


def test_perplexity_single_value():
    logprobs = [-2.0]
    result = compute_perplexity(logprobs)
    assert math.isclose(result, math.exp(2.0))


def test_perplexity_all_zero_logprobs():
    logprobs = [0.0, 0.0, 0.0]
    result = compute_perplexity(logprobs)
    assert math.isclose(result, 1.0)

def test_perplexity_metrics_normal_case():
    logprobs = [-1.0, -2.0, -3.0, -4.0, -5.0]
    perp, bottom_perp = compute_perplexity_metrics(logprobs, bottom_k=3)

    # Expected: perplexity over all, and perplexity over 3 smallest (most negative) logprobs
    expected_full = math.exp(-sum(logprobs) / len(logprobs))
    bottom = sorted(logprobs)[:3]
    expected_bottom = math.exp(-sum(bottom) / len(bottom))

    assert math.isclose(perp, expected_full)
    assert math.isclose(bottom_perp, expected_bottom)


def test_perplexity_metrics_empty_list():
    perp, bottom_perp = compute_perplexity_metrics([], bottom_k=5)
    assert perp == float("inf")
    assert bottom_perp == float("inf")


def test_perplexity_metrics_bottom_k_larger_than_list():
    logprobs = [-1.0, -1.5]
    perp, bottom_perp = compute_perplexity_metrics(logprobs, bottom_k=10)

    assert math.isclose(perp, math.exp(-sum(logprobs) / len(logprobs)))
    assert math.isclose(bottom_perp, math.exp(-sum(logprobs) / len(logprobs)))


def test_perplexity_metrics_identical_logprobs():
    logprobs = [-2.0] * 10
    perp, bottom_perp = compute_perplexity_metrics(logprobs, bottom_k=5)
    expected = math.exp(2.0)
    assert math.isclose(perp, expected)
    assert math.isclose(bottom_perp, expected)