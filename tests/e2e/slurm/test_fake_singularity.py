# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType

import pytest

SHIM = Path(__file__).parent / "shims" / "fake_singularity.py"


def run_shim(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SHIM), *args],
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
        timeout=30,
        check=False,
    )


def _alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def test_exec_runs_command_with_singularityenv_mapped():
    result = run_shim(
        "exec",
        "--nv",
        "--writable-tmpfs",
        "image.sif",
        "sh",
        "-c",
        "echo $GREETING",
        env={"SINGULARITYENV_GREETING": "hello"},
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "hello"


def test_bind_creates_symlink_and_rebind_replaces_it(tmp_path):
    first, second = tmp_path / "first.py", tmp_path / "second.py"
    first.write_text("first")
    second.write_text("second")
    dst = tmp_path / "opt" / "watchdog.py"

    assert run_shim("exec", "--bind", f"{first}:{dst},{tmp_path}", "img", "true").returncode == 0
    assert dst.is_symlink() and dst.read_text() == "first"

    assert run_shim("exec", "-B", f"{second}:{dst}:ro", "img", "true").returncode == 0
    assert dst.read_text() == "second"


def test_bind_replaces_an_existing_directory(tmp_path):
    src = tmp_path / "serving"
    src.mkdir()
    (src / "00-upstreams.conf").write_text("upstream llm {}")
    dst = tmp_path / "etc" / "nginx" / "conf.d"
    dst.mkdir(parents=True)
    (dst / "default.conf").write_text("stale")

    assert run_shim("exec", "-B", f"{src}:{dst}:ro", "img", "true").returncode == 0
    assert sorted(p.name for p in dst.iterdir()) == ["00-upstreams.conf"]


def _load_shim() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fake_singularity", SHIM)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("destination", ["/run", "/run/"])
def test_bind_over_protected_path_is_skipped(tmp_path, monkeypatch, capsys, destination):
    shim = _load_shim()

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("apply_bind touched a protected path")

    # In-process, so that a broken guard fails here instead of deleting the host's /run.
    monkeypatch.setattr(shim.shutil, "rmtree", forbidden)
    monkeypatch.setattr(shim.os, "replace", forbidden)
    monkeypatch.setattr(shim.Path, "symlink_to", forbidden)

    shim.apply_bind(f"{tmp_path}:{destination}")

    assert "skipping bind" in capsys.readouterr().err


def test_concurrent_binds_to_one_destination_both_succeed(tmp_path):
    src = tmp_path / "watchdog.py"
    src.write_text("watchdog")
    dst = tmp_path / "opt" / "watchdog.py"

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(lambda _: run_shim("exec", "--bind", f"{src}:{dst}", "img", "true"), range(8))
        )

    assert [r.returncode for r in results] == [0] * 8, [r.stderr for r in results]
    assert dst.read_text() == "watchdog"


def test_unsupported_option_fails_loudly():
    result = run_shim("exec", "--cleanenv", "img", "true")

    assert result.returncode == 2
    assert "unsupported option: --cleanenv" in result.stderr


def test_unsupported_subcommand_fails_loudly():
    result = run_shim("build", "out.sif", "docker://nginx")

    assert result.returncode == 2
    assert "unsupported subcommand: build" in result.stderr


@pytest.mark.skipif(sys.platform != "linux", reason="reads /proc")
def test_instance_lifecycle_runs_image_as_start_script(tmp_path):
    image = tmp_path / "nginx.sif"
    image.write_text("exec sleep 300\n")
    env = {"FAKE_SINGULARITY_RUN_DIR": str(tmp_path / "run")}

    start = run_shim("instance", "start", "--writable-tmpfs", str(image), "lb_1", env=env)
    assert start.returncode == 0, start.stderr
    pid = int((tmp_path / "run" / "lb_1.pid").read_text())
    assert _alive(pid)

    inside = run_shim("exec", "instance://lb_1", "sh", "-c", "echo inside", env=env)
    assert inside.stdout.strip() == "inside"

    assert run_shim("instance", "stop", "lb_1", env=env).returncode == 0
    deadline = time.monotonic() + 10
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _alive(pid)
    assert not (tmp_path / "run" / "lb_1.pid").exists()


def test_instance_stop_of_unknown_instance_fails(tmp_path):
    result = run_shim(
        "instance", "stop", "missing", env={"FAKE_SINGULARITY_RUN_DIR": str(tmp_path)}
    )

    assert result.returncode == 1
    assert "no instance named missing" in result.stderr
