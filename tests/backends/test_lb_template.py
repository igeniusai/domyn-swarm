from pathlib import Path
from types import SimpleNamespace

import jinja2

TPL_DIR = Path("src/domyn_swarm/templates")


def _cfg(gpu_enabled):
    gx = SimpleNamespace(enabled=gpu_enabled, port=9835, kind="nvidia_smi")
    mon = SimpleNamespace(
        enabled=True,
        route_prefix="/prometheus",
        port=9090,
        mode="binary",
        prometheus_binary="prometheus",
        nginx_exporter_binary="nginx-prometheus-exporter",
        prometheus_image=None,
        nginx_exporter_image=None,
        exporter_port=9113,
        retention="12h",
        gpu_exporter=gx,
    )
    ep = SimpleNamespace(port=9000, monitoring=mon)
    backend = SimpleNamespace(
        endpoint=ep, requires_ray=False, ray_dashboard_port=8265, ray_port=6379
    )
    return SimpleNamespace(backend=backend)


def _render(cfg):
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TPL_DIR),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.get_template("lb.sh.j2").render(
        cfg=cfg,
        job_name="jx",
        dep_jobid=1,
        replicas=2,
        swarm_directory="/swarm",
        collector_script_path="/c.py",
        supervisor_script_path="/s.py",
    )


def test_supervisor_gets_emit_gpu_targets_when_enabled():
    assert _render(_cfg(True)).count("--emit-gpu-targets") == 2


def test_supervisor_no_emit_gpu_targets_when_disabled():
    assert _render(_cfg(False)).count("--emit-gpu-targets") == 0
