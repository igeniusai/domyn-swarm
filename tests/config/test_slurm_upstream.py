from pydantic import ValidationError
import pytest

from domyn_swarm.config.slurm import SlurmEndpointConfig, UpstreamConfig


def test_default_is_least_conn():
    cfg = SlurmEndpointConfig(nginx_image="nginx.sif")
    assert cfg.upstream.strategy == "least_conn"
    assert cfg.upstream.key is None


def test_ip_hash_no_key():
    cfg = UpstreamConfig(strategy="ip_hash")
    assert cfg.strategy == "ip_hash"
    assert cfg.key is None


def test_hash_with_valid_key():
    cfg = UpstreamConfig(strategy="hash", key="$http_x_repo")
    assert cfg.strategy == "hash"
    assert cfg.key == "$http_x_repo"


def test_hash_requires_key():
    with pytest.raises(ValidationError, match="key is required"):
        UpstreamConfig(strategy="hash")


@pytest.mark.parametrize(
    "bad_key",
    ["foo", "; rm -rf /", "$http x repo", "$", "$1foo", "$$x", "http_x_repo"],
)
def test_hash_rejects_invalid_keys(bad_key):
    with pytest.raises(ValidationError, match="nginx variable"):
        UpstreamConfig(strategy="hash", key=bad_key)


def test_non_hash_rejects_key():
    with pytest.raises(ValidationError, match="only valid when strategy='hash'"):
        UpstreamConfig(strategy="least_conn", key="$x")


def test_unknown_strategy_rejected():
    with pytest.raises(ValidationError):
        UpstreamConfig(strategy="round_robin")  # type: ignore[arg-type]
