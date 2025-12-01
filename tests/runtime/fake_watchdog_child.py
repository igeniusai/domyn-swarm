import argparse
import contextlib
import http.server
import os
from pathlib import Path
import socketserver
import sys
import time


def _bump_run_counter(state_file: Path) -> int:
    """
    Increment an integer counter in `state_file` and return the new value.
    If the file does not exist, start from 0.
    """
    state_file = Path(state_file)
    try:
        raw = state_file.read_text().strip()
        count = int(raw) if raw else 0
    except Exception:
        count = 0
    count += 1
    state_file.write_text(str(count))
    return count


class _HealthHandler(http.server.BaseHTTPRequestHandler):
    # Global mode for this process; set by _serve_http()
    mode: str = "healthy"

    def do_GET(self):
        if self.path != "/health":
            self.send_response(404)
            self.end_headers()
            return

        if self.mode == "healthy":
            # 200 OK -> watchdog sees as healthy
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        elif self.mode == "http_unhealthy":
            # 503 -> watchdog sees as UNHEALTHY
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b"UNHEALTHY")
        else:
            # default healthy
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")

    # Silence default noisy logging
    def log_message(self, fmt, *args):
        # suppress logs for tests
        return


def _serve_http(port: int, mode: str) -> None:
    """
    Start a simple HTTP server on /health in the given mode:
      - "healthy"        -> 200 OK
      - "http_unhealthy" -> 503 UNHEALTHY
    """
    handler_cls = _HealthHandler
    handler_cls.mode = mode  # <- this is what do_GET will read

    with (
        socketserver.TCPServer(("127.0.0.1", port), handler_cls) as httpd,
        contextlib.suppress(KeyboardInterrupt),
    ):
        httpd.serve_forever()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--state-file", type=str, required=True)
    args = parser.parse_args(argv)

    state_file = Path(args.state_file)

    mode = os.environ.get("FAKE_CHILD_MODE", "healthy").strip().lower()

    # Increment run counter at each process invocation
    run_no = _bump_run_counter(state_file)

    if mode == "always_fail":
        # Simulate immediate failure without ever becoming healthy
        # Non-zero exit code:
        print(f"[fake-child] run #{run_no}: always_fail", file=sys.stderr)
        return 1

    if mode == "fail_once_then_ok":
        if run_no == 1:
            # First time: fail immediately
            print(f"[fake-child] run #{run_no}: intentional failure", file=sys.stderr)
            return 2  # arbitrary non-zero
        # On subsequent runs, behave like a healthy server
        print(f"[fake-child] run #{run_no}: now healthy", file=sys.stderr)
        # fall through to healthy HTTP server below

    if mode == "always_succeed":
        # No HTTP server needed for the restart-policy test:
        time.sleep(0.2)
        return 0

    if mode == "http_unhealthy":
        # Bind port and always respond 503 on /health
        print(f"[fake-child] run #{run_no}: http_unhealthy on {args.port}", file=sys.stderr)
        _serve_http(args.port, mode="http_unhealthy")
        return 0

    # Default / healthy mode: start a /health server and stay alive
    print(
        f"[fake-child] run #{run_no}: starting healthy HTTP server on {args.port}", file=sys.stderr
    )

    with contextlib.suppress(KeyboardInterrupt):
        _serve_http(args.port, mode="healthy")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
