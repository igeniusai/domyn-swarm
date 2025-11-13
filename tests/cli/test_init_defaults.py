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

from typer.testing import CliRunner
import yaml

import domyn_swarm.cli.init as mod

# -----------------------
# _load_yaml / _save_yaml
# -----------------------


def test_load_yaml_success(tmp_path: Path):
    p = tmp_path / "defaults.yaml"
    data = {"a": {"b": 1}}
    p.write_text(yaml.safe_dump(data))
    out = mod._load_yaml(p)
    assert out == data


def test_load_yaml_missing_returns_empty(tmp_path: Path):
    p = tmp_path / "nonexistent.yaml"
    out = mod._load_yaml(p)
    assert out == {}


def test_load_yaml_bad_yaml_returns_empty(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text(":\n  - this is not valid yaml")
    out = mod._load_yaml(p)
    assert out == {}


def test_save_yaml_creates_parents_and_writes(tmp_path: Path):
    p = tmp_path / "nested" / "dir" / "defaults.yaml"
    payload = {"x": 1, "y": {"z": "ok"}}
    mod._save_yaml(p, payload)
    assert p.exists()
    # Compare by loading (not raw text) to avoid formatting differences
    assert yaml.safe_load(p.read_text()) == payload


# ------
# _get/_set
# ------


def test_get_dotted_returns_values_and_default():
    d = {"a": {"b": {"c": 42}}}
    assert mod._get(d, "a.b.c") == 42
    assert mod._get(d, "a.b.x", default="missing") == "missing"
    assert mod._get(d, "missing", default=None) is None


def test_set_dotted_creates_nested():
    d = {}
    mod._set(d, "a.b.c", 7)
    assert d == {"a": {"b": {"c": 7}}}
    # override existing
    mod._set(d, "a.b.c", 8)
    assert d["a"]["b"]["c"] == 8


# ---------------
# Prompt helpers
# ---------------


def test_prompt_str_enforces_non_empty(monkeypatch):
    answers = iter(["", "   ", " ok  "])
    monkeypatch.setattr(mod.typer, "prompt", lambda label, default="": next(answers))
    # Should loop until non-empty -> returns "ok"
    out = mod._prompt_str("Label")
    assert out == "ok"


def test_prompt_str_allow_empty(monkeypatch):
    monkeypatch.setattr(mod.typer, "prompt", lambda label, default="": "")
    assert mod._prompt_str("Label", allow_empty=True) == ""


def test_prompt_int_validates_and_bounds(monkeypatch):
    answers = iter(["foo", "0", "10"])
    monkeypatch.setattr(mod.typer, "prompt", lambda label, default="": next(answers))
    # first invalid -> error, then 0 is out of bounds (min 1), then 10 OK
    out = mod._prompt_int("Number", default=5, min_v=1, max_v=10)
    assert out == 10


def test_prompt_mem_validates(monkeypatch):
    answers = iter(["blargh", " 16gb "])
    monkeypatch.setattr(mod.typer, "prompt", lambda label, default=None: next(answers))
    out = mod._prompt_mem("Memory", default="8GB")
    assert out == "16GB"


def test_prompt_walltime_validates(monkeypatch):
    answers = iter(["1:2:3", "24:00:00"])
    monkeypatch.setattr(mod.typer, "prompt", lambda label, default=None: next(answers))
    out = mod._prompt_walltime("Wall")
    assert out == "24:00:00"


def test_yesno_passthrough(monkeypatch):
    monkeypatch.setattr(mod.typer, "confirm", lambda label, default=True: False)
    assert mod._yesno("Q?", default=True) is False


# ------------------------------
# _configure_slurm_defaults
# ------------------------------


def test_configure_slurm_defaults_sets_expected_fields(monkeypatch):
    # Order of prompts in _configure_slurm_defaults:
    # image, partition, account, qos, nginx_image,
    # port, poll, cpus, mem, tpc, wall, proxy_buf(confirm), nginx_timeout
    prompt_answers = iter(
        [
            "img.sif",  # image
            "p",  # partition
            "acc",  # account
            "",  # qos (optional -> omitted)
            "nginx.sif",  # nginx_image
            "9001",  # port
            "7",  # poll
            "3",  # cpus
            "32GB",  # mem
            "2",  # threads_per_core
            "12:00:00",  # wall
            "120s",  # nginx_timeout (after confirm below)
        ]
    )

    def fake_prompt(label, default=""):
        # default is sometimes str(int) or string; we ignore it here
        return next(prompt_answers)

    # confirm only used for proxy buffering in this flow
    monkeypatch.setattr(mod.typer, "prompt", fake_prompt)
    monkeypatch.setattr(mod.typer, "confirm", lambda label, default=True: True)

    out = mod._configure_slurm_defaults(existing={})

    # qos omitted since empty
    assert out["slurm"]["image"] == "img.sif"
    assert out["slurm"]["partition"] == "p"
    assert out["slurm"]["account"] == "acc"
    assert "qos" not in out.get("slurm", {})

    ep = out["slurm"]["endpoint"]
    assert ep["nginx_image"] == "nginx.sif"
    assert ep["port"] == 9001
    assert ep["poll_interval"] == 7
    assert ep["cpus_per_task"] == 3
    assert ep["mem"] == "32GB"
    assert ep["threads_per_core"] == 2
    assert ep["wall_time"] == "12:00:00"
    assert ep["enable_proxy_buffering"] is True
    assert ep["nginx_timeout"] == "120s"


# ------------------------------
# _configure_lepton_defaults
# ------------------------------


def test_configure_lepton_defaults_sets_expected_fields(monkeypatch):
    # Order: workspace, resource_shape, node_group, job_image
    prompt_answers = iter(
        [
            "ws123",  # workspace
            "gpu.4xh200",  # resource_shape
            "",  # node_group (optional -> omitted)
            "igeniusai/domyn-swarm:latest",  # job image
        ]
    )

    monkeypatch.setattr(mod.typer, "prompt", lambda label, default="": next(prompt_answers))

    out = mod._configure_lepton_defaults(existing={})
    assert out["lepton"]["workspace_id"] == "ws123"
    assert out["lepton"]["endpoint"]["resource_shape"] == "gpu.4xh200"
    assert "node_group" not in out["lepton"]["endpoint"]
    assert out["lepton"]["job"]["image"] == "igeniusai/domyn-swarm:latest"


# ------------------------------
# Typer command: create_defaults
# ------------------------------


def test_create_defaults_aborts_on_existing_without_force(monkeypatch, tmp_path):
    # Prepare existing file
    out = tmp_path / "defaults.yaml"
    out.write_text("a: 1\n")

    # Avoid interactive prompts: the very first prompt would be overwrite (handled via _yesno)
    monkeypatch.setattr(mod, "_yesno", lambda *a, **k: False)

    runner = CliRunner()
    result = runner.invoke(mod.init_app, ["defaults", "-o", str(out)])
    assert result.exit_code != 0
    # File should be unchanged
    assert yaml.safe_load(out.read_text()) == {"a": 1}


def test_create_defaults_writes_when_confirmed(monkeypatch, tmp_path):
    out = tmp_path / "defaults.yaml"

    # Stub the interactive flows to avoid prompts:
    # - Select Slurm only
    # - Preview confirm True
    yesno_answers = iter(
        [
            True,  # want_slurm?
            False,  # want_lepton?
            True,
        ]
    )  # confirm write?
    monkeypatch.setattr(mod, "_yesno", lambda *a, **k: next(yesno_answers))

    # Avoid the internal prompts by stubbing the flows
    monkeypatch.setattr(
        mod,
        "_configure_slurm_defaults",
        lambda existing: {"slurm": {"image": "img.sif"}},
    )
    monkeypatch.setattr(
        mod,
        "_configure_lepton_defaults",
        lambda existing: {"lepton": {"workspace_id": "ws"}},
    )

    # Avoid reading any existing YAML
    monkeypatch.setattr(mod, "_load_yaml", lambda p: {})

    runner = CliRunner()
    result = runner.invoke(mod.init_app, ["-o", str(out), "--force"])
    assert result.exit_code == 0
    assert out.exists()
    data = yaml.safe_load(out.read_text())
    assert data == {"slurm": {"image": "img.sif"}}


def test_create_defaults_handles_no_sections_selected(monkeypatch, tmp_path):
    out = tmp_path / "defaults.yaml"

    # Choose nothing → abort
    yesno_answers = iter(
        [
            False,  # want_slurm?
            False,
        ]
    )  # want_lepton?
    monkeypatch.setattr(mod, "_yesno", lambda *a, **k: next(yesno_answers))
    # Ensure load returns empty
    monkeypatch.setattr(mod, "_load_yaml", lambda p: {})

    runner = CliRunner()
    result = runner.invoke(mod.init_app, ["defaults", "-o", str(out), "--force"])
    assert result.exit_code != 0
    assert not out.exists()
