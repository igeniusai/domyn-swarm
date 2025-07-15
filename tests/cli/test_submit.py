from typer.testing import CliRunner
from domyn_swarm.cli.submit import submit_app
from utils import print_cli_debug, extract_rich_text

runner = CliRunner()


def test_submit_script_requires_config_or_state(tmp_path):

    # Create a dummy script file
    script_path = tmp_path / "script.py"
    script_path.write_text("print('Hello, Domyn Swarm!')\n")

    result = runner.invoke(submit_app, ["script", str(script_path)])
    assert result.exit_code != 0


def test_submit_script_no_such_file(tmp_path):
    result = runner.invoke(submit_app, ["script", "non_existent_script.py"])

    assert result.exit_code != 0
    assert "No such file or directory" in extract_rich_text(result.stderr)


def test_submit_job_mutual_exclusion(tmp_path):

    config_path = tmp_path / "config.yaml"
    config_path.write_text("dummy_config: true\n")

    result = runner.invoke(
        submit_app,
        [
            "job",
            "domyn_swarm.jobs:ChatCompletionJob",
            "--input",
            "dummy.parquet",
            "--output",
            "dummy_out.parquet",
            "--config",
            str(config_path),
            "--state",
            "swarm_123.json",
        ],
    )
    assert result.exit_code != 0
