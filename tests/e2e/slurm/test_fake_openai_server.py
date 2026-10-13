# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator
import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

SERVER = Path(__file__).parent / "shims" / "fake_openai_server.py"
MODEL = "e2e/fake-model"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _get(url: str) -> tuple[int, bytes]:
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.status, response.read()


def _post_chat(base: str, model: str) -> tuple[int, dict]:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": "ping"}]})
    request = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=body.encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


@pytest.fixture
def server() -> Iterator[tuple[subprocess.Popen[bytes], str]]:
    port = _free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            str(SERVER),
            "serve",
            MODEL,
            "--tensor-parallel-size",
            "1",
            "--max-model-len",
            "64",
            "--port",
            str(port),
        ]
    )
    base = f"http://127.0.0.1:{port}"
    # Interpreter start-up on a loaded shared filesystem can take many seconds.
    deadline = time.monotonic() + 30
    while True:
        try:
            _get(f"{base}/health")
            break
        except OSError as error:
            if process.poll() is not None:
                raise AssertionError(f"server exited with {process.returncode}") from error
            if time.monotonic() > deadline:
                process.kill()
                raise AssertionError("server did not answer /health within 30s") from error
            time.sleep(0.1)
    yield process, base
    if process.poll() is None:
        process.kill()
        process.wait()


def test_unknown_vllm_arguments_are_ignored(server):
    _, base = server

    status, body = _get(f"{base}/health")

    assert status == 200
    assert body == b""


def test_models_lists_the_served_model(server):
    _, base = server

    _, body = _get(f"{base}/v1/models")

    assert [m["id"] for m in json.loads(body)["data"]] == [MODEL]


def test_chat_completion_returns_openai_shape(server):
    _, base = server

    status, body = _post_chat(base, MODEL)

    assert status == 200
    assert body["object"] == "chat.completion"
    assert body["model"] == MODEL
    assert body["choices"][0]["message"]["role"] == "assistant"


def test_chat_completion_for_another_model_is_404(server):
    _, base = server

    status, body = _post_chat(base, "other-model")

    assert status == 404
    assert "other-model" in body["error"]["message"]


def test_sigterm_stops_the_server_cleanly(server):
    process, _ = server

    process.send_signal(signal.SIGTERM)

    assert process.wait(timeout=5) == 0


def test_missing_serve_subcommand_is_rejected():
    result = subprocess.run(
        [sys.executable, str(SERVER), "bench", MODEL], capture_output=True, text=True, check=False
    )

    assert result.returncode != 0
    assert "usage: vllm serve" in result.stderr
