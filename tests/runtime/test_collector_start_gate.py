# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Behaviour of the collector startup helpers used by the lb job."""

from pathlib import Path
import socket
import subprocess
import sys
import textwrap

import jinja2
import pytest

from .helpers import get_free_port

pytestmark = pytest.mark.integration

TEMPLATES = Path(__file__).resolve().parents[2] / "src" / "domyn_swarm" / "templates"


def _collector_helpers() -> str:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATES),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.get_template("_collector.sh.j2").render()


def _run_driver(tmp_path: Path, body: str, name: str = "driver.sh") -> subprocess.CompletedProcess:
    script = tmp_path / name
    script.write_text(f"{_collector_helpers()}\n{textwrap.dedent(body)}")
    return subprocess.run(
        ["bash", script.as_posix()],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
    )


def _driver_preamble(tmp_path: Path, port: int) -> str:
    return f"""
        COLLECTOR_LOG="{tmp_path}/collector.log"
        COLLECTOR_READY_FILE="{tmp_path}/collector.ready"
        COLLECTOR_HOST="localhost"
        COLLECTOR_PORT={port}
        COLLECTOR_START_TIMEOUT_S=20
        : >"$COLLECTOR_LOG"
    """


def _real_collector_cmd(tmp_path: Path) -> str:
    return f"""
        collector_cmd() {{
            "{sys.executable}" -m domyn_swarm.runtime.collector \\
              --db "{tmp_path}/watchdog.db" \\
              --host 127.0.0.1 \\
              --port "$COLLECTOR_PORT" \\
              --ready-file "$COLLECTOR_READY_FILE"
        }}
    """


def test_gate_fails_when_the_collector_cannot_bind_its_port(tmp_path):
    with socket.socket() as blocker:
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        port = blocker.getsockname()[1]

        proc = _run_driver(
            tmp_path,
            f"""
            set -eu
            {_driver_preamble(tmp_path, port)}
            {_real_collector_cmd(tmp_path)}
            collector_start_or_die
            echo GATE-OK
            """,
        )

    assert proc.returncode != 0
    assert "GATE-OK" not in proc.stdout
    assert "collector failed to start" in proc.stderr
    # The operator gets the reason, not just the fact.
    assert "cannot bind" in proc.stderr.lower()
    assert str(port) in proc.stderr


def test_gate_passes_once_the_collector_is_listening(tmp_path):
    port = get_free_port()
    proc = _run_driver(
        tmp_path,
        f"""
        set -eu
        {_driver_preamble(tmp_path, port)}
        {_real_collector_cmd(tmp_path)}
        collector_start_or_die
        echo GATE-OK
        kill "$COLLECTOR_PID"
        """,
    )
    assert proc.returncode == 0, proc.stderr
    assert "GATE-OK" in proc.stdout
    assert f"collector listening on localhost:{port}" in proc.stdout
