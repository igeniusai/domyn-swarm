#!/usr/bin/env python
"""
Minimal stub for `ray` CLI used in watchdog integration tests.

Supports:
  - `ray status` -> exit 0 (Ray alive)
  - `ray list nodes --format=json` -> configurable cluster via env:

    FAKE_RAY_NODES          (default: 2)
    FAKE_RAY_GPUS_PER_NODE  (default: 1.0)

You can tweak env vars per-test to model "capacity OK" vs "capacity insufficient".
"""

import json
import os
import sys


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
        num_alive = int(os.getenv("FAKE_RAY_NODES", "2"))
        gpus_per_node = float(os.getenv("FAKE_RAY_GPUS_PER_NODE", "1.0"))

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
