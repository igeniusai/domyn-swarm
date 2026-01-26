#!/usr/bin/env python
"""
Minimal stub for `ray` CLI used in watchdog integration tests.

Supports:
  - `ray status` -> exit 0 (Ray alive)
  - `ray list nodes --format=json` -> configurable cluster via env:

    FAKE_RAY_NODES          (default: 2)
    FAKE_RAY_GPUS_PER_NODE  (default: 1.0)
    FAKE_RAY_STATE_FILE     (optional JSON file; overrides the env vars per invocation)

You can tweak env vars per-test to model "capacity OK" vs "capacity insufficient".
"""

import json
import os
import sys


def _read_state_from_file(path: str) -> tuple[int, float] | None:
    """Read (alive_nodes, gpus_per_node) from a JSON state file."""
    try:
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        data = json.loads(raw or "{}")
        alive_nodes = int(data.get("alive_nodes", 2))
        gpus_per_node = float(data.get("gpus_per_node", 1.0))
        return alive_nodes, gpus_per_node
    except OSError:
        return None
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def _get_cluster_state() -> tuple[int, float]:
    """Return (alive_nodes, gpus_per_node) for the fake Ray cluster."""
    state_file = (os.getenv("FAKE_RAY_STATE_FILE") or "").strip()
    if state_file:
        from_file = _read_state_from_file(state_file)
        if from_file is not None:
            return from_file

    num_alive = int(os.getenv("FAKE_RAY_NODES", "2"))
    gpus_per_node = float(os.getenv("FAKE_RAY_GPUS_PER_NODE", "1.0"))
    return num_alive, gpus_per_node


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        return 0

    # We model only the subset we need for the watchdog:
    #   ray status
    #   ray list nodes --format=json
    sub = argv[0]

    # --- ray status ---------------------------------------------------------
    if sub == "status":
        # Treat as "Ray cluster is alive"
        print("fake-ray: status OK")
        return 0

    # --- ray list nodes --format=json --------------------------------------
    if sub == "list" and len(argv) >= 2 and argv[1] == "nodes" and "--format=json" in argv:
        num_alive, gpus_per_node = _get_cluster_state()

        nodes = [
            {
                "state": "ALIVE",
                "resources_total": {"GPU": gpus_per_node},
            }
            for _ in range(num_alive)
        ]
        print(json.dumps(nodes))
        return 0

    # Any other subcommand: just succeed quietly.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
