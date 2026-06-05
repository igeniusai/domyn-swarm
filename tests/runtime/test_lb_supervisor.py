from pathlib import Path

from domyn_swarm.runtime.lb_supervisor import read_head_files, render_upstreams


def _write_heads(serving_dir: Path, addrs: dict[int, str]) -> None:
    for replica_id, addr in addrs.items():
        (serving_dir / f"replica-{replica_id}.head").write_text(addr + "\n")


def test_read_head_files_sorted_by_replica_id(tmp_path: Path):
    _write_heads(tmp_path, {2: "h2:9000", 0: "h0:9000", 1: "h1:9000"})
    assert read_head_files(tmp_path) == ["h0:9000", "h1:9000", "h2:9000"]


def test_read_head_files_empty_when_none(tmp_path: Path):
    assert read_head_files(tmp_path) == []


def test_render_upstreams_no_ray(tmp_path: Path):
    _write_heads(tmp_path, {0: "h0:9000", 1: "h1:9000"})
    conf = render_upstreams(tmp_path, ray_enabled=False, ray_dashboard_port=8265, ray_port=6379)
    assert "upstream llm {" in conf
    assert "least_conn;" in conf
    assert "server h0:9000 max_fails=2 fail_timeout=10s;" in conf
    assert "server h1:9000 max_fails=2 fail_timeout=10s;" in conf
    assert "upstream ray" not in conf


def test_render_upstreams_with_ray_uses_replica0_host(tmp_path: Path):
    _write_heads(tmp_path, {0: "rayhost:9000"})
    conf = render_upstreams(tmp_path, ray_enabled=True, ray_dashboard_port=8265, ray_port=6379)
    assert "upstream ray {" in conf
    assert "server rayhost:8265;" in conf
    assert "upstream ray_control {" in conf
    assert "server rayhost:6379;" in conf


def test_render_upstreams_ray_enabled_empty_dir_has_no_ray_block(tmp_path: Path):
    conf = render_upstreams(tmp_path, ray_enabled=True, ray_dashboard_port=8265, ray_port=6379)
    assert "upstream llm {" in conf
    assert "upstream ray" not in conf
