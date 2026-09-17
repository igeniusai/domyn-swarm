# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Commands that build a swarm from a config must report bad paths like `up`.

`job submit`, `job submit-script` and `pool` all enter a swarm, so they inherit
the same check. Without handling they would surface it as a raw traceback.
"""

from click import unstyle
import pytest
from typer.testing import CliRunner
import yaml

from domyn_swarm.cli.main import app

runner = CliRunner()

PREFLIGHT_EXIT_CODE = 2


@pytest.fixture
def bad_config(tmp_path):
    """A config whose vLLM image does not exist."""
    nginx_image = tmp_path / "nginx.sif"
    nginx_image.touch()
    path = tmp_path / "swarm.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "nemotron",
                "model": "nvidia/Nemotron-Nano-9B-v2",
                "image": "/shared/images/vllm.sif",
                "backend": {
                    "type": "slurm",
                    "account": "a",
                    "partition": "p",
                    "qos": "q",
                    "endpoint": {"nginx_image": str(nginx_image)},
                },
            }
        )
    )
    return path


@pytest.fixture
def script_file(tmp_path):
    path = tmp_path / "script.py"
    path.write_text("print('hi')\n")
    return path


def test_job_submit_script_reports_the_bad_path(bad_config, script_file, disable_autoupgrade):
    result = runner.invoke(app, ["job", "submit-script", str(script_file), "-c", str(bad_config)])
    stderr = unstyle(result.stderr)

    assert result.exit_code == PREFLIGHT_EXIT_CODE
    assert "/shared/images/vllm.sif" in stderr
    assert "Traceback" not in stderr


def test_job_submit_reports_the_bad_path(bad_config, tmp_path, disable_autoupgrade):
    input_parquet = tmp_path / "in.parquet"
    input_parquet.touch()

    result = runner.invoke(
        app,
        [
            "job",
            "submit",
            "domyn_swarm.jobs:ChatCompletionJob",
            "-c",
            str(bad_config),
            "--input",
            str(input_parquet),
            "--output",
            str(tmp_path / "out.parquet"),
        ],
    )
    stderr = unstyle(result.stderr)

    assert result.exit_code == PREFLIGHT_EXIT_CODE
    assert "/shared/images/vllm.sif" in stderr
    assert "Traceback" not in stderr


def test_pool_reports_the_bad_path(bad_config, tmp_path, disable_autoupgrade):
    pool_config = tmp_path / "pool.yaml"
    pool_config.write_text(
        yaml.safe_dump(
            {"pool": [{"name": "nemotron", "config_path": str(bad_config), "replicas": 1}]}
        )
    )

    result = runner.invoke(app, ["pool", "pool", str(pool_config)])
    stderr = unstyle(result.stderr)

    assert result.exit_code == PREFLIGHT_EXIT_CODE
    assert "/shared/images/vllm.sif" in stderr
    assert "Traceback" not in stderr
