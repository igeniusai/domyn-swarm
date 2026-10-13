# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Stand-in for ``vllm serve`` with the OpenAI endpoints that the e2e tests use.

It accepts the full ``vllm serve`` command line that the Slurm template builds
and ignores every option except the model, host, port, and served model name.
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import signal
import sys
import threading
import time

REPLY = "pong"


def parse_args(argv: list[str]) -> argparse.Namespace:
    if not argv or argv[0] != "serve":
        sys.exit("usage: vllm serve MODEL [options] --port PORT")
    parser = argparse.ArgumentParser(prog="vllm serve")
    parser.add_argument("model")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--served-model-name", default=None)
    args, _ = parser.parse_known_args(argv[1:])
    return args


def make_handler(model_name: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, b"")
            elif self.path == "/v1/models":
                self._send_json(
                    200,
                    {"object": "list", "data": [{"id": model_name, "object": "model"}]},
                )
            else:
                self._send_json(404, {"error": {"message": f"no route {self.path}"}})

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self._send_json(404, {"error": {"message": f"no route {self.path}"}})
                return
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length) or b"{}")
            requested = request.get("model")
            if requested != model_name:
                self._send_json(
                    404, {"error": {"message": f"The model `{requested}` does not exist."}}
                )
                return
            self._send_json(
                200,
                {
                    "id": "chatcmpl-e2e",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": REPLY},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )

        def _send(self, status: int, body: bytes, content_type: str = "text/plain") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, payload: dict) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(args.served_model_name or args.model)
    )
    # shutdown() blocks until serve_forever() returns, so it cannot run on the
    # thread that is inside serve_forever().
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown).start())
    print(f"fake vllm serving {args.model} on {args.host}:{args.port}", flush=True)
    server.serve_forever()
    server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
