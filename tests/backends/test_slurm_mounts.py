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

from pathlib import Path

import jinja2
import pytest

import domyn_swarm
from domyn_swarm.config.slurm import SlurmConfig
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


def _make_cfg(mounts):
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


def test_mounts_field_defaults_to_empty_list():
    cfg = SlurmConfig(partition="p", account="a", qos="q")
    assert cfg.mounts == []


def test_relative_mount_source_rejected():
    with pytest.raises(ValueError):
        SlurmConfig(partition="p", account="a", qos="q", mounts=["relative/path"])


def test_empty_mount_entry_rejected():
    with pytest.raises(ValueError):
        SlurmConfig(partition="p", account="a", qos="q", mounts=["  "])


def test_too_many_colon_segments_rejected():
    with pytest.raises(ValueError):
        SlurmConfig(partition="p", account="a", qos="q", mounts=["/a:/b:ro:extra"])
