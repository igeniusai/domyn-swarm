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

from types import SimpleNamespace

from domyn_swarm.backends.serving.srun_builder import SrunCommandBuilder


def _fake_cfg(mem="32GB", cpus=8):
    """Make a minimal cfg with .endpoint.mem / .endpoint.cpus_per_task."""
    endpoint = SimpleNamespace(mem=mem, cpus_per_task=cpus)
    return SimpleNamespace(endpoint=endpoint)


def test_build_basic_no_env_no_mail_no_extra():
    cfg = _fake_cfg(mem="16GB", cpus=2)
    b = SrunCommandBuilder(cfg=cfg, jobid=123, nodelist="nodeA")
    exe = ["python", "-m", "domyn_swarm.jobs.run", "--help"]

    cmd = b.build(exe)

    # Must start with srun and core flags
    assert cmd[:7] == [
        "srun",
        "--jobid=123",
        "--nodelist=nodeA",
        "--ntasks=1",
        "--overlap",
        "--mem=16GB",
        "--cpus-per-task=2",
    ]

    # With no env, we should get a plain --export=ALL
    assert "--export=ALL" in cmd
    # No mail flags by default
    assert not any(arg.startswith("--mail-user=") for arg in cmd)
    assert "--mail-type=END,FAIL" not in cmd

    # Executable is appended at the end in order
    assert cmd[-len(exe) :] == exe


def test_build_with_env_appends_export_all_and_vars():
    cfg = _fake_cfg()
    b = SrunCommandBuilder(cfg=cfg, jobid=99, nodelist="n1").with_env(
        {"ENDPOINT": "http://x:9000", "MODEL": "mistral"}
    )
    exe = ["bash", "-lc", "echo ok"]
    cmd = b.build(exe, ntasks=3)

    # ntasks honored
    assert "--ntasks=3" in cmd

    # Find the export flag and check its contents (order is not guaranteed)
    export_args = [a for a in cmd if a.startswith("--export=ALL")]
    assert len(export_args) == 1
    exp = export_args[0]
    assert exp.startswith("--export=ALL,")
    # Split and check presence of both key=val pairs
    suffix = exp.split(",", 1)[1]
    parts = set(suffix.split(","))
    assert "ENDPOINT=http://x:9000" in parts
    assert "MODEL=mistral" in parts


def test_build_with_mail_adds_mail_flags():
    cfg = _fake_cfg()
    b = SrunCommandBuilder(cfg=cfg, jobid=7, nodelist="n2").with_mail("u@example.com")
    exe = ["whoami"]
    cmd = b.build(exe)

    assert "--mail-user=u@example.com" in cmd
    assert "--mail-type=END,FAIL" in cmd
    # Still contains export (either ALL or with vars)
    assert any(arg.startswith("--export=ALL") for arg in cmd)


def test_build_with_extra_args_before_exe():
    cfg = _fake_cfg()
    extra = ["--gres=gpu:1", "--qos=high"]
    b = (
        SrunCommandBuilder(cfg=cfg, jobid=1, nodelist="n3")
        .with_extra_args(extra)
        .with_env({"FOO": "BAR"})
        .with_mail("me@x.y")
    )
    exe = ["python", "-c", "print(1)"]
    cmd = b.build(exe)

    # Ensure extra args exist and come before the exe part
    for a in extra:
        assert a in cmd
    # Executable is the tail
    assert cmd[-len(exe) :] == exe

    # The last occurrence of any extra arg must be before the exe index
    last_extra_idx = max(cmd.index(a) for a in extra)
    exe_start_idx = len(cmd) - len(exe)
    assert last_extra_idx < exe_start_idx


def test_export_all_without_env_is_exact_flag():
    cfg = _fake_cfg()
    b = SrunCommandBuilder(cfg=cfg, jobid=5, nodelist="n5")
    cmd = b.build(["/bin/true"])
    # There should be EXACTLY one export flag and it must be "--export=ALL"
    export_args = [a for a in cmd if a.startswith("--export=")]
    assert export_args == ["--export=ALL"]
