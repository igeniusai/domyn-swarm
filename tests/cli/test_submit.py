# Copyright 2025 Domyn
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

from typer.testing import CliRunner

from domyn_swarm.cli.job import job_app

runner = CliRunner()


def test_submit_script_requires_config_or_state(tmp_path):
    # Create a dummy script file
    script_path = tmp_path / "script.py"
    script_path.write_text("print('Hello, Domyn Swarm!')\n")

    result = runner.invoke(job_app, ["script", str(script_path)])
    assert result.exit_code != 0


def test_submit_script_no_such_file(tmp_path):
    result = runner.invoke(job_app, ["script", "non_existent_script.py"])

    assert result.exit_code != 0


def test_submit_job_mutual_exclusion(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dummy_config: true\n")

    result = runner.invoke(
        job_app,
        [
            "submit",
            "domyn_swarm.jobs:ChatCompletionJob",
            "--input",
            "dummy.parquet",
            "--output",
            "dummy_out.parquet",
            "--config",
            str(config_path),
            "--name",
            "my-swarm-name",
        ],
    )
    assert result.exit_code != 0
