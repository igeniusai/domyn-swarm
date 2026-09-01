# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import subprocess

import jinja2
import pytest

import domyn_swarm
from domyn_swarm.config.slurm import SlurmConfig, SlurmEndpointConfig
from domyn_swarm.helpers.io import is_folder, path_exists
from domyn_swarm.runtime import watchdog_args as watchdog_args_mod


def _render_template(cfg) -> str:
    """Render the real llm_swarm.sh.j2 template the way SlurmDriver does."""
    import domyn_swarm.runtime.watchdog as watchdog_mod

    template_path = cfg.backend.template_path
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_path.parent),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.get_template(template_path.name).render(
        cfg=cfg,
        job_name="test_job",
        path_exists=path_exists,
        is_folder=is_folder,
        cuda_visible_devices="0,1",
        swarm_directory="/tmp/swarm",
        watchdog_script_path=Path(watchdog_mod.__file__).resolve().as_posix(),
        build_watchdog_args=watchdog_args_mod.build_watchdog_args,
        args_to_str=watchdog_args_mod.args_to_str,
        dswarm_agent_version=domyn_swarm.__version__,
    )


def _make_cfg(mounts, system_mounts=None):
    from domyn_swarm.config.swarm import DomynLLMSwarmConfig

    return DomynLLMSwarmConfig(
        name="gpt4",
        image="/path/to/vllm.sif",
        model="gpt-4",
        replicas=2,
        backend=SlurmConfig(
            type="slurm",
            partition="debug",
            account="test_account",
            qos="test_qos",
            mounts=mounts,
            system_mounts=system_mounts or [],
            endpoint=SlurmEndpointConfig(nginx_image="/path/to/nginx.sif"),
        ).model_dump(),
    )


def test_extra_mounts_rendered_in_template():
    cfg = _make_cfg(["/data/models", "/host/path:/container/path:ro"])
    rendered = _render_template(cfg)
    assert "MOUNTS=$MOUNTS,/data/models" in rendered
    assert "MOUNTS=$MOUNTS,/host/path:/container/path:ro" in rendered


def test_no_extra_mounts_renders_no_extra_lines():
    cfg = _make_cfg([])
    rendered = _render_template(cfg)
    # The default base MOUNTS export is always present, but no appended user mounts.
    assert 'export MOUNTS="' in rendered
    appended = [
        line
        for line in rendered.splitlines()
        if line.strip().startswith("MOUNTS=$MOUNTS,")
        and "LOCAL_RAY_LOGS" not in line
        and "cfg.model" not in line
    ]
    assert appended == []


def _slurm_config(**kwargs) -> SlurmConfig:
    """Build a SlurmConfig with an explicit endpoint so no site defaults are needed."""
    return SlurmConfig(
        partition="p",
        account="a",
        qos="q",
        endpoint=SlurmEndpointConfig(nginx_image="/path/to/nginx.sif"),
        **kwargs,
    )


def test_mounts_field_defaults_to_empty_list():
    cfg = _slurm_config()
    assert cfg.mounts == []


def test_relative_mount_source_rejected():
    with pytest.raises(ValueError):
        _slurm_config(mounts=["relative/path"])


def test_empty_mount_entry_rejected():
    with pytest.raises(ValueError):
        _slurm_config(mounts=["  "])


def test_too_many_colon_segments_rejected():
    with pytest.raises(ValueError):
        _slurm_config(mounts=["/a:/b:ro:extra"])


def test_system_mounts_rendered_as_existence_checked_binds():
    cfg = _make_cfg([], system_mounts=["/opt/site/stack", "/host/lib:/usr/local/lib:ro"])
    rendered = _render_template(cfg)
    assert 'add_optional_mount "/opt/site/stack"' in rendered
    assert 'add_optional_mount "/host/lib:/usr/local/lib:ro"' in rendered


def test_no_site_specific_path_hardcoded_in_templates():
    """No cluster-specific path may live in the shipped templates."""
    template_dir = Path(domyn_swarm.__file__).resolve().parent / "templates"
    for template in template_dir.glob("*.j2"):
        assert "/leonardo" not in template.read_text(), f"site path hardcoded in {template.name}"


def test_base_mounts_have_no_system_mounts_by_default():
    cfg = _make_cfg([])
    rendered = _render_template(cfg)
    # Only the built-in optional bind (the interconnect devices) is checked.
    optional = [
        line
        for line in rendered.splitlines()
        if line.startswith("add_optional_mount") and not line.endswith("{")
    ]
    assert optional == ["add_optional_mount /dev/infiniband"]


def test_optional_mount_helper_skips_missing_paths(tmp_path):
    """The rendered helper must skip absent host paths instead of failing."""
    cfg = _make_cfg([], system_mounts=[tmp_path.as_posix(), "/definitely/not/here"])
    rendered = _render_template(cfg)
    start = rendered.index("add_optional_mount() {")
    end = rendered.index('add_optional_mount "/definitely/not/here"') + len(
        'add_optional_mount "/definitely/not/here"'
    )
    script = 'MOUNTS="base"\n' + rendered[start:end] + '\necho "$MOUNTS"'
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout
    assert f"base,{tmp_path.as_posix()}" in out
    assert "/definitely/not/here" not in out.splitlines()[-1]


def test_system_mounts_field_defaults_to_empty_list():
    cfg = _slurm_config()
    assert cfg.system_mounts == []


def test_system_mounts_resolved_from_defaults_file(monkeypatch, tmp_path):
    from domyn_swarm.config import defaults as defaults_mod

    defaults_file = tmp_path / "defaults.yaml"
    defaults_file.write_text("slurm:\n  system_mounts:\n    - /site/prod/opt\n")
    monkeypatch.setenv("DOMYN_SWARM_DEFAULTS", str(defaults_file))
    defaults_mod.reload_defaults_cache()
    try:
        assert _slurm_config().system_mounts == ["/site/prod/opt"]
    finally:
        defaults_mod.reload_defaults_cache()


def test_relative_system_mount_source_rejected():
    with pytest.raises(ValueError):
        _slurm_config(system_mounts=["relative/path"])
