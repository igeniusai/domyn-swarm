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

import subprocess
import types

import pytest

from domyn_swarm.helpers.slurm import (
    get_job_status,
    is_job_running,
)


def test_get_job_status_returns_trimmed_status(monkeypatch):
    # Fake subprocess.run for the list-argv call
    def fake_run(argv, capture_output, text, check):
        # verify shape of command; not strictly required
        assert isinstance(argv, list)
        assert "squeue" in argv[0]
        # Surrounding whitespace/newline should be stripped
        return types.SimpleNamespace(stdout="  RUNNING \n", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    status = get_job_status(12345)
    assert status == "RUNNING"


def test_get_job_status_raises_runtime_error_on_failure(monkeypatch):
    def fake_run(argv, capture_output, text, check):
        # Simulate squeue error with check=True by raising CalledProcessError
        raise subprocess.CalledProcessError(
            returncode=1, cmd=argv, output="", stderr="boom"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError) as ei:
        get_job_status(999)
    assert "999" in str(ei.value)
    # Optional: ensure original exception chained
    assert isinstance(ei.value.__cause__, subprocess.CalledProcessError)


def test_is_job_running_true_when_id_present(monkeypatch):
    captured = {}

    def fake_run(cmd, shell, text, capture_output):
        # The function uses a pipeline; ensure it invoked with shell=True
        captured["cmd"] = cmd
        captured["shell"] = shell
        assert shell is True
        # Simulate output lines (already awk'd/tailed to only job IDs)
        return types.SimpleNamespace(stdout="11111\n22222\n33333\n", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert is_job_running("22222") is True
    # sanity: verify the expected base command was used
    assert "squeue --me --states=R" in captured["cmd"]


def test_is_job_running_false_when_id_absent(monkeypatch):
    def fake_run(cmd, shell, text, capture_output):
        return types.SimpleNamespace(stdout="11111\n", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert is_job_running("99999") is False


def test_is_job_running_false_on_empty_output(monkeypatch):
    def fake_run(cmd, shell, text, capture_output):
        # No running jobs
        return types.SimpleNamespace(stdout="", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert is_job_running("12345") is False
