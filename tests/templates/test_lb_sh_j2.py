from pathlib import Path
from types import SimpleNamespace

import jinja2
import pytest

from domyn_swarm.config.slurm import UpstreamConfig

TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "src" / "domyn_swarm" / "templates"


def _render(upstream: UpstreamConfig) -> str:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)),
        keep_trailing_newline=True,
    )
    cfg = SimpleNamespace(
        image="img.sif",
        wait_endpoint_s=1800,
        backend=SimpleNamespace(
            account="acct",
            partition="part",
            qos="qos",
            mail_user=None,
            requires_ray=False,
            ray_port=6379,
            ray_dashboard_port=8265,
            endpoint=SimpleNamespace(
                cpus_per_task=1,
                mem="1GB",
                threads_per_core=1,
                wall_time="1:00:00",
                enable_proxy_buffering=True,
                nginx_timeout="60s",
                port=9000,
                nginx_image="nginx.sif",
                upstream=upstream,
            ),
        ),
    )
    return env.get_template("lb.sh.j2").render(
        cfg=cfg,
        job_name="job",
        dep_jobid="42",
        replicas=1,
        swarm_directory="/tmp/swarm",
        collector_script_path="/tmp/collector.py",
    )


def test_default_emits_least_conn():
    out = _render(UpstreamConfig())
    assert 'echo "    least_conn;"' in out
    assert "ip_hash" not in out
    assert "hash $" not in out


def test_explicit_least_conn():
    out = _render(UpstreamConfig(strategy="least_conn"))
    assert 'echo "    least_conn;"' in out


def test_ip_hash_emits_directive():
    out = _render(UpstreamConfig(strategy="ip_hash"))
    assert 'echo "    ip_hash;"' in out
    assert "least_conn" not in out


def test_hash_emits_single_quoted_directive():
    out = _render(UpstreamConfig(strategy="hash", key="$http_x_repo"))
    # Single-quoted bash echo: $http_x_repo passes through to nginx unexpanded
    assert "echo '    hash $http_x_repo consistent;'" in out
    # Make sure the literal $ is preserved (no \$ escaping artifact)
    assert "\\$http_x_repo" not in out
    assert "least_conn" not in out


@pytest.mark.parametrize("key", ["$arg_repo", "$request_uri", "$http_x_my_header"])
def test_hash_supports_other_nginx_vars(key):
    out = _render(UpstreamConfig(strategy="hash", key=key))
    assert f"echo '    hash {key} consistent;'" in out
