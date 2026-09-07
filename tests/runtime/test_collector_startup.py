# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import socket
import subprocess
import sys
import time

import pytest

from .helpers import get_free_port, wait_for_port

pytestmark = pytest.mark.integration


def _collector_cmd(db_path: Path, port: int, ready_file: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "domyn_swarm.runtime.collector",
        "--db",
        db_path.as_posix(),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ready-file",
        ready_file.as_posix(),
    ]


def test_collector_fails_with_a_clear_error_when_its_port_is_taken(tmp_path):
    ready_file = tmp_path / "collector.ready"

    with socket.socket() as blocker:
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        port = blocker.getsockname()[1]

        proc = subprocess.run(
            _collector_cmd(tmp_path / "watchdog.db", port, ready_file),
            capture_output=True,
            text=True,
            timeout=60,
        )

    assert proc.returncode == 1, proc.stderr
    assert "cannot bind" in proc.stderr.lower()
    assert str(port) in proc.stderr
    assert "Traceback" not in proc.stderr
    assert not ready_file.exists()


def test_collector_writes_ready_file_with_the_bound_port(tmp_path):
    port = get_free_port()
    ready_file = tmp_path / "collector.ready"

    proc = subprocess.Popen(
        _collector_cmd(tmp_path / "watchdog.db", port, ready_file),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.time() + 15.0
        while time.time() < deadline and not ready_file.exists():
            assert proc.poll() is None, proc.communicate()[1]
            time.sleep(0.05)

        assert ready_file.exists()
        assert ready_file.read_text().strip() == str(port)
        # The file only appears once the socket really accepts connections.
        wait_for_port("127.0.0.1", port, timeout=5.0)
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    assert not ready_file.exists()
