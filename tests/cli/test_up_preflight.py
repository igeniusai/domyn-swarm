# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""``domyn-swarm up`` must reject unusable config paths before submitting."""

from click import unstyle
import pytest
from typer.testing import CliRunner
import yaml

import domyn_swarm.cli.main as main

runner = CliRunner()

PREFLIGHT_EXIT_CODE = 2


@pytest.fixture
def config_file(tmp_path):
    """Write a Slurm swarm config whose image path does not exist."""
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


def test_up_exits_before_submitting_when_a_path_is_missing(
    config_file, mocker, disable_autoupgrade
):
    swarm_cls = mocker.patch.object(main, "DomynLLMSwarm")

    result = runner.invoke(main.app, ["up", "-c", str(config_file)])
    stderr = unstyle(result.stderr)

    # Exit 2 is also click's usage-error code, so pin it with the report text.
    assert result.exit_code == PREFLIGHT_EXIT_CODE
    assert "Nothing was submitted" in stderr
    swarm_cls.assert_not_called()


def test_up_names_the_field_value_and_nearest_existing_parent(
    config_file, mocker, disable_autoupgrade
):
    mocker.patch.object(main, "DomynLLMSwarm")

    result = runner.invoke(main.app, ["up", "-c", str(config_file)])
    stderr = unstyle(result.stderr)

    assert "image" in stderr
    assert "/shared/images/vllm.sif" in stderr
    assert "deepest existing parent: /" in stderr
    assert "Nothing was submitted" in stderr


def test_up_reports_nothing_on_stdout_so_command_substitution_stays_clean(
    config_file, mocker, disable_autoupgrade
):
    mocker.patch.object(main, "DomynLLMSwarm")

    result = runner.invoke(main.app, ["up", "-c", str(config_file)])

    assert result.stdout == ""


def test_skip_preflight_hands_the_opt_out_to_the_swarm(config_file, mocker, disable_autoupgrade):
    """Skipping the CLI check is not enough: __enter__ runs the same check."""
    fake_swarm = mocker.MagicMock()
    fake_swarm.__enter__.return_value = fake_swarm
    fake_swarm.__exit__.return_value = None
    swarm_cls = mocker.patch.object(main, "DomynLLMSwarm", return_value=fake_swarm)

    result = runner.invoke(main.app, ["up", "-c", str(config_file), "--skip-preflight"])

    assert result.exit_code == 0, result.output
    assert swarm_cls.call_args.kwargs["preflight"] is False


def test_preflight_is_on_by_default_for_the_swarm(tmp_path, mocker, disable_autoupgrade):
    image = tmp_path / "vllm.sif"
    image.touch()
    nginx_image = tmp_path / "nginx.sif"
    nginx_image.touch()
    path = tmp_path / "ok.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "nemotron",
                "model": "nvidia/Nemotron-Nano-9B-v2",
                "image": str(image),
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
    fake_swarm = mocker.MagicMock()
    fake_swarm.__enter__.return_value = fake_swarm
    fake_swarm.__exit__.return_value = None
    swarm_cls = mocker.patch.object(main, "DomynLLMSwarm", return_value=fake_swarm)

    result = runner.invoke(main.app, ["up", "-c", str(path)])

    assert result.exit_code == 0, result.output
    assert swarm_cls.call_args.kwargs["preflight"] is True


def test_up_proceeds_when_every_path_exists(tmp_path, mocker, disable_autoupgrade):
    image = tmp_path / "vllm.sif"
    image.touch()
    nginx_image = tmp_path / "nginx.sif"
    nginx_image.touch()

    path = tmp_path / "swarm.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "nemotron",
                "model": "nvidia/Nemotron-Nano-9B-v2",
                "image": str(image),
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

    fake_swarm = mocker.MagicMock()
    fake_swarm.__enter__.return_value = fake_swarm
    fake_swarm.__exit__.return_value = None
    swarm_cls = mocker.patch.object(main, "DomynLLMSwarm", return_value=fake_swarm)

    result = runner.invoke(main.app, ["up", "-c", str(path)])

    assert result.exit_code == 0, result.output
    swarm_cls.assert_called_once()
