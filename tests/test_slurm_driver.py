import pytest
from unittest.mock import patch, MagicMock, ANY
from domyn_swarm.core.slurm_driver import SlurmDriver
from domyn_swarm.models.swarm import DomynLLMSwarmConfig


@pytest.fixture
def dummy_config(tmp_path):
    return DomynLLMSwarmConfig(
        model="gpt-4",
        replicas=2,
        template_path=tmp_path / "template.sh.j2",
        log_directory=tmp_path / "logs",
        home_directory=tmp_path,
    )


@pytest.fixture
def slurm_driver(dummy_config):
    return SlurmDriver(cfg=dummy_config)


@patch("domyn_swarm.core.slurm_driver.subprocess.check_output")
@patch("domyn_swarm.core.slurm_driver.jinja2.Environment.get_template")
def test_submit_replicas(mock_get_template, mock_check_output, slurm_driver, tmp_path):
    mock_template = MagicMock()
    mock_template.render.return_value = "#!/bin/bash\necho test"
    mock_get_template.return_value = mock_template
    mock_check_output.return_value = "12345;dummy"

    job_id = slurm_driver.submit_replicas("test_job")
    assert job_id == 12345
    mock_get_template.assert_called()
    mock_check_output.assert_called()


@patch("domyn_swarm.core.slurm_driver.subprocess.check_output")
@patch("domyn_swarm.core.slurm_driver.jinja2.Environment.get_template")
def test_submit_lb(mock_get_template, mock_check_output, slurm_driver):
    mock_template = MagicMock()
    mock_template.render.return_value = "#!/bin/bash\necho lb"
    mock_get_template.return_value = mock_template
    mock_check_output.return_value = "67890"

    job_id = slurm_driver.submit_lb("test_lb_job", 12345)
    assert job_id == 67890
    mock_get_template.assert_called()
    mock_check_output.assert_called_with(
        [
            "sbatch",
            "--parsable",
            "--dependency",
            "after:12345",
            "--export",
            "DEP_JOBID=12345",
            ANY,
        ],
        text=True,
    )


@patch("domyn_swarm.core.slurm_driver.subprocess.check_output")
def test_get_node_from_jobid(mock_check_output, slurm_driver):
    mock_check_output.side_effect = ["nodespec\n", "node001\nnode002\n"]
    node = slurm_driver.get_node_from_jobid(11111)
    assert node == "node001"
