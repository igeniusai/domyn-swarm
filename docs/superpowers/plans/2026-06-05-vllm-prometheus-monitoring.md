# vLLM / nginx Prometheus Monitoring — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional Prometheus-based monitoring of vLLM replicas and nginx load balancing via a sidecar on the Slurm LB node, consumable by grafatui/Grafana, while refactoring the oversized `lb.sh.j2` into a leaner, testable Python supervisor.

**Architecture:** Phase 0 verifies the one feasibility-sensitive mechanism (cross-container SIGHUP reload). Phase 1 is a behavior-preserving refactor: the nginx config-generation/reload logic moves out of the bash `lb.sh.j2` into a Python `lb_supervisor.py` (mirroring the existing `collector.py` pattern), with a static server config template and a dynamic upstreams file. Phase 2 adds the monitoring feature on top: a `MonitoringConfig` block, Prometheus + nginx-exporter sidecars, nginx `/prometheus/` + `/nginx_status` locations, a `domyn-swarm monitor` command, and a bundled Grafana dashboard JSON.

**Tech Stack:** Python 3.10+, pydantic v2, Typer, Jinja2, Singularity/Apptainer, Prometheus, nginx-prometheus-exporter, grafatui (external). Tests: pytest (+ pytest-asyncio), hatchling build.

**Design source:** `docs/superpowers/specs/2026-06-05-vllm-prometheus-monitoring-design.md`

**Conventions (from AGENTS.md):** `uv run` for everything; Google-style docstrings on all new `src/` functions/classes; ruff (line-length 100, py310) + pyright must pass; mirror source tree in `tests/`; Conventional Commits.

---

## File structure

**Phase 1 (refactor):**
- Create: `src/domyn_swarm/runtime/lb_supervisor.py` — pure config generators + watch loop + SIGHUP reload + CLI. Single responsibility: keep nginx's dynamic upstreams in sync with `replica-*.head` files.
- Create: `src/domyn_swarm/templates/nginx_server.conf.j2` — static `server {}` block + locations (rendered once into `serving/10-server.conf`).
- Modify: `src/domyn_swarm/templates/nginx.conf.j2` — repurpose as the LB **main** nginx config (`pid /run/nginx.pid;`, `events`, `http { include conf.d }`).
- Modify: `src/domyn_swarm/templates/lb.sh.j2` — shrink to: sbatch header, start collector, wait for replicas, render static configs, start nginx instance, start supervisor, traps.
- Modify: `src/domyn_swarm/backends/serving/slurm_driver.py` — pass `supervisor_script_path` (and render the static server conf) into the LB job.
- Test: `tests/runtime/test_lb_supervisor.py`, `tests/backends/test_lb_template_render.py`.

**Phase 2 (monitoring):**
- Modify: `src/domyn_swarm/config/slurm.py` — add `MonitoringConfig`, attach as `SlurmEndpointConfig.monitoring`.
- Modify: `src/domyn_swarm/runtime/lb_supervisor.py` — also emit `targets.json`.
- Modify: `src/domyn_swarm/templates/nginx_server.conf.j2` — add `/nginx_status` + `/prometheus/` locations (gated).
- Modify: `src/domyn_swarm/templates/lb.sh.j2` — launch prometheus + nginx-exporter sidecars (gated), extend traps.
- Modify: `src/domyn_swarm/backends/serving/slurm_driver.py` — write `prometheus.yml` into serving dir.
- Create: `src/domyn_swarm/cli/monitor.py` — `domyn-swarm monitor` command.
- Modify: `src/domyn_swarm/cli/main.py` — register the command.
- Create: `src/domyn_swarm/data/dashboards/vllm.json` — bundled Grafana dashboard.
- Test: `tests/config/test_monitoring_config.py`, `tests/cli/test_monitor.py`, `tests/data/test_dashboard.py`, plus additions to the Phase 1 test files.

---

## Phase 0 — Feasibility spike (manual, do first)

### Task 0: Verify cross-container SIGHUP reload on the target cluster

**Why:** The supervisor runs inside `PYTHON_IMG` and must reload nginx (running in a separate singularity instance) by sending `SIGHUP` to the nginx master PID. This relies on singularity instances sharing the host PID namespace by default. Confirm before building on it.

- [ ] **Step 1: Start an nginx instance with a known pid path**

On a compute/login node with singularity and the nginx image:

```bash
mkdir -p /tmp/sigtest/run /tmp/sigtest/conf.d /tmp/sigtest/logs
cat > /tmp/sigtest/nginx.conf <<'EOF'
pid /run/nginx.pid;
events { worker_connections 1024; }
http { include /etc/nginx/conf.d/*.conf; }
EOF
cat > /tmp/sigtest/conf.d/10-server.conf <<'EOF'
server { listen 18080; location / { return 200 "v1\n"; } }
EOF
singularity instance start --writable-tmpfs \
  -B /tmp/sigtest/nginx.conf:/etc/nginx/nginx.conf:ro \
  -B /tmp/sigtest/conf.d:/etc/nginx/conf.d:ro \
  -B /tmp/sigtest/run:/run \
  "$NGINX_IMG" sigtest
singularity exec instance://sigtest nginx
```

Expected: `curl -s localhost:18080` → `v1`; `/tmp/sigtest/run/nginx.pid` exists and contains the nginx master PID (a host-visible PID).

- [ ] **Step 2: From a *different* container, edit config and SIGHUP the master**

```bash
echo 'server { listen 18080; location / { return 200 "v2\n"; } }' > /tmp/sigtest/conf.d/10-server.conf
singularity exec -B /tmp/sigtest "$PYTHON_IMG" \
  python3 -c "import os, signal; pid=int(open('/tmp/sigtest/run/nginx.pid').read()); os.kill(pid, signal.SIGHUP)"
sleep 1
curl -s localhost:18080
```

Expected: `v2` — confirms a process in `PYTHON_IMG` can SIGHUP nginx running in another instance.

- [ ] **Step 3: Record the outcome**

If it works: proceed with the plan as written.
If it does NOT work (e.g. cluster forces PID-namespace isolation): use the documented fallback — keep a minimal bash reload loop in `lb.sh.j2` that runs `singularity exec instance://"$INSTANCE_NAME" nginx -s reload` when the supervisor signals a change via a touch-file (`$SERVING_DIR/.reload`), and make the supervisor `touch` that file instead of calling `os.kill`. All other tasks are unchanged. Note the chosen path at the top of `lb_supervisor.py`.

- [ ] **Step 4: Clean up**

```bash
singularity instance stop sigtest; rm -rf /tmp/sigtest
```

---

## Phase 1 — Behavior-preserving LB refactor

> Goal of this phase: identical runtime behavior to today, but the nginx generation/reload logic lives in testable Python and `lb.sh.j2` is much smaller. No monitoring yet.

### Task 1: Pure upstreams-config generator

**Files:**
- Create: `src/domyn_swarm/runtime/lb_supervisor.py`
- Test: `tests/runtime/test_lb_supervisor.py`

- [ ] **Step 1: Write failing tests for `read_head_files` and `render_upstreams`**

```python
# tests/runtime/test_lb_supervisor.py
from pathlib import Path

from domyn_swarm.runtime.lb_supervisor import read_head_files, render_upstreams


def _write_heads(serving_dir: Path, addrs: dict[int, str]) -> None:
    for replica_id, addr in addrs.items():
        (serving_dir / f"replica-{replica_id}.head").write_text(addr)


def test_read_head_files_sorted_by_replica_id(tmp_path: Path):
    _write_heads(tmp_path, {2: "h2:9000", 0: "h0:9000", 1: "h1:9000"})
    assert read_head_files(tmp_path) == ["h0:9000", "h1:9000", "h2:9000"]


def test_read_head_files_empty_when_none(tmp_path: Path):
    assert read_head_files(tmp_path) == []


def test_render_upstreams_no_ray(tmp_path: Path):
    _write_heads(tmp_path, {0: "h0:9000", 1: "h1:9000"})
    conf = render_upstreams(tmp_path, ray_enabled=False, ray_dashboard_port=8265, ray_port=6379)
    assert "upstream llm {" in conf
    assert "least_conn;" in conf
    assert "server h0:9000 max_fails=2 fail_timeout=10s;" in conf
    assert "server h1:9000 max_fails=2 fail_timeout=10s;" in conf
    assert "upstream ray" not in conf


def test_render_upstreams_with_ray_uses_replica0_host(tmp_path: Path):
    _write_heads(tmp_path, {0: "rayhost:9000"})
    conf = render_upstreams(tmp_path, ray_enabled=True, ray_dashboard_port=8265, ray_port=6379)
    assert "upstream ray {" in conf
    assert "server rayhost:8265;" in conf
    assert "upstream ray_control {" in conf
    assert "server rayhost:6379;" in conf
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/runtime/test_lb_supervisor.py -v`
Expected: FAIL — `ModuleNotFoundError: domyn_swarm.runtime.lb_supervisor`.

- [ ] **Step 3: Implement the module skeleton + generators**

Port the exact logic from the current `lb.sh.j2` `generate_nginx_conf()` (upstream block + ray upstreams derived from `replica-0.head`).

```python
# src/domyn_swarm/runtime/lb_supervisor.py
#!/usr/bin/env python3
"""Domyn-Swarm load-balancer supervisor.

Keeps nginx's dynamic upstreams (and, when monitoring is enabled, the Prometheus
``targets.json``) in sync with the ``replica-*.head`` files that each vLLM replica
writes into the serving directory. Runs inside the same Python image as the
collector and is launched by the LB sbatch job (see ``templates/lb.sh.j2``).

Reload mechanism: the supervisor sends ``SIGHUP`` to the nginx master process
(PID read from ``<serving_dir>/../run/nginx.pid``). Singularity instances share
the host PID namespace by default, so this reaches nginx running in a separate
instance. See Phase 0 spike in the implementation plan.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys
import time


def read_head_files(serving_dir: Path) -> list[str]:
    """Return replica ``host:port`` addresses, ordered by replica id.

    Args:
        serving_dir: Directory containing ``replica-<id>.head`` files, each
            holding a single ``host:port`` line.

    Returns:
        Addresses sorted ascending by the integer replica id encoded in the
        filename. Empty if no head files exist yet.
    """
    serving_dir = Path(serving_dir)
    entries: list[tuple[int, str]] = []
    for f in serving_dir.glob("replica-*.head"):
        try:
            replica_id = int(f.stem.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        addr = f.read_text().strip()
        if addr:
            entries.append((replica_id, addr))
    entries.sort(key=lambda e: e[0])
    return [addr for _, addr in entries]


def render_upstreams(
    serving_dir: Path,
    *,
    ray_enabled: bool,
    ray_dashboard_port: int,
    ray_port: int,
) -> str:
    """Render the nginx ``upstream`` blocks for the current replicas.

    Reproduces the upstream generation previously embedded in ``lb.sh.j2``.

    Args:
        serving_dir: Directory containing ``replica-*.head`` files.
        ray_enabled: Whether to also emit the Ray dashboard/control upstreams
            (derived from ``replica-0.head``).
        ray_dashboard_port: Ray dashboard port for the ``ray`` upstream.
        ray_port: Ray client port for the ``ray_control`` upstream.

    Returns:
        nginx configuration text defining the ``llm`` upstream (and Ray
        upstreams when enabled).
    """
    addrs = read_head_files(serving_dir)
    lines = ["upstream llm {", "  least_conn;"]
    for addr in addrs:
        lines.append(f"  server {addr} max_fails=2 fail_timeout=10s;")
    lines.append("}")

    if ray_enabled and addrs:
        ray_host = addrs[0].split(":", 1)[0]
        lines += [
            "upstream ray {",
            f"  server {ray_host}:{ray_dashboard_port};",
            "  keepalive 8;",
            "}",
            "upstream ray_control {",
            f"  server {ray_host}:{ray_port};",
            "  keepalive 8;",
            "}",
        ]
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/runtime/test_lb_supervisor.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/domyn_swarm/runtime/lb_supervisor.py tests/runtime/test_lb_supervisor.py
git commit -m "feat(runtime): add lb_supervisor upstreams generator"
```

### Task 2: Atomic write + change detection

**Files:**
- Modify: `src/domyn_swarm/runtime/lb_supervisor.py`
- Test: `tests/runtime/test_lb_supervisor.py`

- [ ] **Step 1: Write failing tests**

```python
# append to tests/runtime/test_lb_supervisor.py
from domyn_swarm.runtime.lb_supervisor import write_if_changed


def test_write_if_changed_creates_and_reports_true(tmp_path: Path):
    target = tmp_path / "00-upstreams.conf"
    assert write_if_changed(target, "abc\n") is True
    assert target.read_text() == "abc\n"


def test_write_if_changed_noop_when_same(tmp_path: Path):
    target = tmp_path / "00-upstreams.conf"
    write_if_changed(target, "abc\n")
    assert write_if_changed(target, "abc\n") is False


def test_write_if_changed_no_partial_temp_left(tmp_path: Path):
    target = tmp_path / "00-upstreams.conf"
    write_if_changed(target, "abc\n")
    assert list(tmp_path.glob("*.tmp")) == []
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/runtime/test_lb_supervisor.py -k write_if_changed -v`
Expected: FAIL — `ImportError: cannot import name 'write_if_changed'`.

- [ ] **Step 3: Implement**

```python
# add to src/domyn_swarm/runtime/lb_supervisor.py
import os


def write_if_changed(target: Path, content: str) -> bool:
    """Atomically write ``content`` to ``target`` only if it differs.

    Writes to a sibling temp file and ``os.replace`` so readers never observe a
    partial file.

    Args:
        target: Destination path.
        content: Text to write.

    Returns:
        True if the file was created or its content changed; False if the
        existing content already matched.
    """
    target = Path(target)
    if target.exists() and target.read_text() == content:
        return False
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(content)
    os.replace(tmp, target)
    return True
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/runtime/test_lb_supervisor.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(runtime): add atomic write_if_changed to lb_supervisor"
```

### Task 3: nginx SIGHUP reload helper

**Files:**
- Modify: `src/domyn_swarm/runtime/lb_supervisor.py`
- Test: `tests/runtime/test_lb_supervisor.py`

- [ ] **Step 1: Write failing tests**

```python
# append to tests/runtime/test_lb_supervisor.py
import signal as _signal

from domyn_swarm.runtime import lb_supervisor as lbs


def test_reload_nginx_sends_sighup(tmp_path: Path, monkeypatch):
    pid_file = tmp_path / "nginx.pid"
    pid_file.write_text("4321\n")
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(lbs.os, "kill", lambda pid, sig: calls.append((pid, sig)))
    assert lbs.reload_nginx(pid_file) is True
    assert calls == [(4321, _signal.SIGHUP)]


def test_reload_nginx_missing_pidfile_returns_false(tmp_path: Path):
    assert lbs.reload_nginx(tmp_path / "absent.pid") is False


def test_reload_nginx_dead_process_returns_false(tmp_path: Path, monkeypatch):
    pid_file = tmp_path / "nginx.pid"
    pid_file.write_text("4321")

    def _raise(pid, sig):
        raise ProcessLookupError

    monkeypatch.setattr(lbs.os, "kill", _raise)
    assert lbs.reload_nginx(pid_file) is False
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/runtime/test_lb_supervisor.py -k reload_nginx -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'reload_nginx'`.

- [ ] **Step 3: Implement**

```python
# add to src/domyn_swarm/runtime/lb_supervisor.py


def reload_nginx(pid_file: Path) -> bool:
    """Trigger an nginx config reload by sending SIGHUP to its master process.

    Args:
        pid_file: Path to the nginx master pid file (e.g. ``/run/nginx.pid``).

    Returns:
        True if SIGHUP was delivered; False if the pid file is missing,
        unreadable, or the process no longer exists. Never raises — reload
        failures must not crash the supervisor.
    """
    pid_file = Path(pid_file)
    try:
        pid = int(pid_file.read_text().strip())
    except (FileNotFoundError, ValueError):
        return False
    try:
        os.kill(pid, signal.SIGHUP)
    except (ProcessLookupError, PermissionError, OSError) as e:
        print(f"lb_supervisor: nginx reload failed: {e!r}", file=sys.stderr)
        return False
    return True
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/runtime/test_lb_supervisor.py -v`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(runtime): add nginx SIGHUP reload helper"
```

### Task 4: One reconcile step (generate upstreams → reload on change)

**Files:**
- Modify: `src/domyn_swarm/runtime/lb_supervisor.py`
- Test: `tests/runtime/test_lb_supervisor.py`

- [ ] **Step 1: Write failing test**

```python
# append to tests/runtime/test_lb_supervisor.py
from dataclasses import dataclass


def test_reconcile_writes_and_reloads_on_change(tmp_path: Path, monkeypatch):
    serving = tmp_path / "serving"
    serving.mkdir()
    run = tmp_path / "run"
    run.mkdir()
    (run / "nginx.pid").write_text("999")
    (serving / "replica-0.head").write_text("h0:9000")

    reloads: list[bool] = []
    monkeypatch.setattr(lbs, "reload_nginx", lambda pid_file: reloads.append(True) or True)

    opts = lbs.SupervisorOptions(
        serving_dir=serving,
        pid_file=run / "nginx.pid",
        ray_enabled=False,
        ray_dashboard_port=8265,
        ray_port=6379,
    )
    # First reconcile: file created -> reload
    assert lbs.reconcile_once(opts) is True
    assert (serving / "00-upstreams.conf").read_text().count("server h0:9000") == 1
    assert reloads == [True]
    # Second reconcile, no change -> no reload
    assert lbs.reconcile_once(opts) is False
    assert reloads == [True]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/runtime/test_lb_supervisor.py -k reconcile -v`
Expected: FAIL — `AttributeError: ... 'SupervisorOptions'`.

- [ ] **Step 3: Implement**

```python
# add to src/domyn_swarm/runtime/lb_supervisor.py
from dataclasses import dataclass

UPSTREAMS_FILENAME = "00-upstreams.conf"


@dataclass
class SupervisorOptions:
    """Runtime options for the LB supervisor reconcile loop.

    Attributes:
        serving_dir: Directory holding ``replica-*.head`` and generated conf.d files.
        pid_file: nginx master pid file used for SIGHUP reloads.
        ray_enabled: Whether to emit Ray upstreams.
        ray_dashboard_port: Ray dashboard port.
        ray_port: Ray client port.
    """

    serving_dir: Path
    pid_file: Path
    ray_enabled: bool
    ray_dashboard_port: int
    ray_port: int


def reconcile_once(opts: SupervisorOptions) -> bool:
    """Regenerate the upstreams file from head files and reload nginx if changed.

    Args:
        opts: Supervisor options.

    Returns:
        True if the upstreams file changed (and a reload was attempted), else False.
    """
    conf = render_upstreams(
        opts.serving_dir,
        ray_enabled=opts.ray_enabled,
        ray_dashboard_port=opts.ray_dashboard_port,
        ray_port=opts.ray_port,
    )
    changed = write_if_changed(opts.serving_dir / UPSTREAMS_FILENAME, conf)
    if changed:
        reload_nginx(opts.pid_file)
    return changed
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/runtime/test_lb_supervisor.py -v`
Expected: PASS (11 tests).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(runtime): add reconcile_once for lb_supervisor"
```

### Task 5: Watch loop + CLI entry point

**Files:**
- Modify: `src/domyn_swarm/runtime/lb_supervisor.py`
- Test: `tests/runtime/test_lb_supervisor.py`

- [ ] **Step 1: Write failing tests for arg parsing**

```python
# append to tests/runtime/test_lb_supervisor.py
def test_parse_args_builds_options():
    opts, once, interval = lbs.parse_args([
        "--serving-dir", "/srv",
        "--pid-file", "/run/nginx.pid",
        "--ray-enabled",
        "--ray-dashboard-port", "8265",
        "--ray-port", "6379",
        "--interval", "5",
        "--once",
    ])
    assert opts.serving_dir == Path("/srv")
    assert opts.pid_file == Path("/run/nginx.pid")
    assert opts.ray_enabled is True
    assert opts.ray_dashboard_port == 8265
    assert opts.ray_port == 6379
    assert interval == 5
    assert once is True


def test_parse_args_defaults_ray_disabled():
    opts, once, interval = lbs.parse_args(["--serving-dir", "/srv", "--pid-file", "/p"])
    assert opts.ray_enabled is False
    assert once is False
    assert interval == 5
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/runtime/test_lb_supervisor.py -k parse_args -v`
Expected: FAIL — `AttributeError: ... 'parse_args'`.

- [ ] **Step 3: Implement loop + CLI**

```python
# add to src/domyn_swarm/runtime/lb_supervisor.py


def run_loop(opts: SupervisorOptions, interval: int) -> int:
    """Continuously reconcile upstreams with head files until terminated.

    Args:
        opts: Supervisor options.
        interval: Seconds to sleep between reconcile passes.

    Returns:
        Process exit code (0 on clean shutdown via SIGTERM/SIGINT).
    """
    stop = {"flag": False}

    def _handle(signum, frame):
        print(f"lb_supervisor: received signal {signum}, stopping", file=sys.stderr)
        stop["flag"] = True

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    print(
        f"lb_supervisor: watching {opts.serving_dir} (interval={interval}s, "
        f"ray_enabled={opts.ray_enabled})",
        file=sys.stderr,
    )
    while not stop["flag"]:
        try:
            reconcile_once(opts)
        except Exception as e:  # never let a transient error kill the loop
            print(f"lb_supervisor: reconcile error: {e!r}", file=sys.stderr)
        for _ in range(interval):
            if stop["flag"]:
                break
            time.sleep(1)
    return 0


def parse_args(argv: list[str] | None = None) -> tuple[SupervisorOptions, bool, int]:
    """Parse CLI arguments into supervisor options.

    Args:
        argv: Argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        A tuple of (options, run_once, interval_seconds).
    """
    p = argparse.ArgumentParser(description="Domyn-Swarm LB supervisor.")
    p.add_argument("--serving-dir", required=True)
    p.add_argument("--pid-file", required=True)
    p.add_argument("--ray-enabled", action="store_true")
    p.add_argument("--ray-dashboard-port", type=int, default=8265)
    p.add_argument("--ray-port", type=int, default=6379)
    p.add_argument("--interval", type=int, default=5)
    p.add_argument("--once", action="store_true")
    a = p.parse_args(argv)
    opts = SupervisorOptions(
        serving_dir=Path(a.serving_dir),
        pid_file=Path(a.pid_file),
        ray_enabled=a.ray_enabled,
        ray_dashboard_port=a.ray_dashboard_port,
        ray_port=a.ray_port,
    )
    return opts, a.once, a.interval


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the LB supervisor."""
    opts, once, interval = parse_args(argv)
    if once:
        reconcile_once(opts)
        return 0
    return run_loop(opts, interval)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/runtime/test_lb_supervisor.py -v`
Expected: PASS (13 tests).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(runtime): add lb_supervisor watch loop and CLI"
```

### Task 6: Static nginx main + server config templates

**Files:**
- Modify: `src/domyn_swarm/templates/nginx.conf.j2` (repurpose as LB main config)
- Create: `src/domyn_swarm/templates/nginx_server.conf.j2`
- Test: `tests/backends/test_lb_template_render.py`

- [ ] **Step 1: Write failing render test**

```python
# tests/backends/test_lb_template_render.py
from pathlib import Path

import jinja2

TEMPLATES = Path("src/domyn_swarm/templates")


def _env() -> jinja2.Environment:
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATES),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def test_nginx_main_sets_run_pid_and_includes_confd():
    out = _env().get_template("nginx.conf.j2").render()
    assert "pid /run/nginx.pid;" in out
    assert "include /etc/nginx/conf.d/*.conf;" in out


def test_server_conf_has_llm_proxy_and_health(monkeypatch):
    # minimal cfg stand-in for the template
    class EP:
        port = 9000
        nginx_timeout = "60s"
        enable_proxy_buffering = True

    class Backend:
        requires_ray = False
        ray_dashboard_port = 8265
        ray_port = 6379
        endpoint = EP()

    class Cfg:
        backend = Backend()

    out = _env().get_template("nginx_server.conf.j2").render(cfg=Cfg())
    assert "listen 9000;" in out
    assert "proxy_pass http://llm;" in out
    assert "location = /health" in out
    assert "proxy_buffer_size 64k;" in out
    assert "/ray/" not in out  # ray disabled


def test_server_conf_includes_ray_when_enabled():
    class EP:
        port = 9000
        nginx_timeout = "60s"
        enable_proxy_buffering = False

    class Backend:
        requires_ray = True
        ray_dashboard_port = 8265
        ray_port = 6379
        endpoint = EP()

    class Cfg:
        backend = Backend()

    out = _env().get_template("nginx_server.conf.j2").render(cfg=Cfg())
    assert "location ^~ /ray/" in out
    assert "proxy_buffering off;" in out
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/backends/test_lb_template_render.py -v`
Expected: FAIL — `nginx_server.conf.j2` not found / assertions fail.

- [ ] **Step 3: Rewrite `nginx.conf.j2` as the LB main config**

```jinja
{# src/domyn_swarm/templates/nginx.conf.j2 — LB main nginx config #}
pid /run/nginx.pid;

events {
    worker_connections 100000;
}

http {
    include /etc/nginx/conf.d/*.conf;
}
```

- [ ] **Step 4: Create `nginx_server.conf.j2`** (port the `server {}` block from the current `lb.sh.j2` heredoc verbatim; upstreams now come from `00-upstreams.conf`)

```jinja
{# src/domyn_swarm/templates/nginx_server.conf.j2 — static server block (conf.d/10-server.conf) #}
server {
  listen {{ cfg.backend.endpoint.port }};
  location / {
    proxy_pass http://llm;
    proxy_connect_timeout       {{ cfg.backend.endpoint.nginx_timeout }};
    proxy_send_timeout          {{ cfg.backend.endpoint.nginx_timeout }};
    proxy_read_timeout          {{ cfg.backend.endpoint.nginx_timeout }};
    send_timeout                {{ cfg.backend.endpoint.nginx_timeout }};
    proxy_set_header Connection "";
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    {% if cfg.backend.endpoint.enable_proxy_buffering %}
    proxy_buffer_size 64k;
    proxy_buffers 16 512k;
    proxy_busy_buffers_size 1m;
    proxy_max_temp_file_size 0;
    client_max_body_size 50m;
    client_body_buffer_size 1m;
    {% else %}
    proxy_buffering off;
    proxy_request_buffering off;
    {% endif %}
    proxy_next_upstream error timeout http_502 http_504;
    proxy_next_upstream_tries 3;
  }
  location = /health {
    proxy_pass http://llm/health;
    proxy_connect_timeout 2s;
    proxy_read_timeout 2s;
    proxy_cache_bypass $http_upgrade;
  }
  {% if cfg.backend.requires_ray %}
  location ^~ /ray/ {
    proxy_pass              http://ray/;
    proxy_read_timeout      300s;
    proxy_connect_timeout   60s;
    proxy_set_header Host   $host;
    proxy_set_header X-Real-IP $remote_addr;
  }
  location = /ray/dashboard {
    return 301 /ray/dashboard/;
  }
  {% endif %}
}
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/backends/test_lb_template_render.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "refactor(templates): split LB nginx into main + static server config"
```

### Task 7: Render the static server conf at submit time

**Files:**
- Modify: `src/domyn_swarm/backends/serving/slurm_driver.py:103-139` (`submit_endpoint`)
- Test: `tests/backends/test_slurm_driver.py`

- [ ] **Step 1: Write failing test**

```python
# append to tests/backends/test_slurm_driver.py
from pathlib import Path
from unittest.mock import patch


@patch("domyn_swarm.backends.serving.slurm_driver.subprocess.check_output")
def test_submit_endpoint_writes_static_server_conf(mock_check_output, slurm_driver, tmp_path):
    mock_check_output.return_value = "55555"
    swarm_dir = tmp_path / "swarms" / "s1"
    (swarm_dir / "serving").mkdir(parents=True)
    slurm_driver.submit_endpoint("job", 12345, 2, str(swarm_dir))
    server_conf = swarm_dir / "serving" / "10-server.conf"
    assert server_conf.exists()
    assert "proxy_pass http://llm;" in server_conf.read_text()
```

(If `slurm_driver`'s `cfg.backend.endpoint.port` is needed, the existing `slurm_driver` fixture in this file already provides a config; reuse it.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/backends/test_slurm_driver.py -k static_server_conf -v`
Expected: FAIL — `10-server.conf` missing.

- [ ] **Step 3: Implement — render server conf + pass supervisor path**

In `submit_endpoint`, after rendering `lb.sh.j2`, render the server config to the serving dir and add `supervisor_script_path` to the `lb.sh.j2` render context. Add the import near `collector_mod`:

```python
# top of slurm_driver.py, alongside the collector import
from domyn_swarm.runtime import collector as collector_mod
from domyn_swarm.runtime import lb_supervisor as supervisor_mod
```

```python
# inside submit_endpoint, replace the lb.sh.j2 render block:
serving_dir = Path(swarm_directory) / "serving"
serving_dir.mkdir(parents=True, exist_ok=True)

# Static nginx server block (dynamic upstreams come from the supervisor).
server_conf = env.get_template("nginx_server.conf.j2").render(cfg=self.cfg)
(serving_dir / "10-server.conf").write_text(server_conf)

lb_script_txt = env.get_template("lb.sh.j2").render(
    cfg=self.cfg,
    job_name=job_name,
    dep_jobid=dep_jobid,
    replicas=replicas,
    swarm_directory=swarm_directory,
    collector_script_path=Path(collector_mod.__file__).resolve().as_posix(),
    supervisor_script_path=Path(supervisor_mod.__file__).resolve().as_posix(),
)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/backends/test_slurm_driver.py -v`
Expected: PASS (existing tests + new one). The existing `test_submit_endpoint` mocks `get_template`, so its `render.return_value` applies to both templates — adjust that mock if it asserts call counts (it doesn't; it only checks the returned job id).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(slurm): render static server conf and pass supervisor path"
```

### Task 8: Slim `lb.sh.j2` to use the supervisor

**Files:**
- Modify: `src/domyn_swarm/templates/lb.sh.j2`
- Test: `tests/backends/test_lb_template_render.py`

- [ ] **Step 1: Write failing render test**

```python
# append to tests/backends/test_lb_template_render.py
class _EP:
    cpus_per_task = 32
    mem = "16GB"
    threads_per_core = 1
    wall_time = "24:00:00"
    port = 9000
    nginx_image = "/img/nginx.sif"
    nginx_timeout = "60s"
    enable_proxy_buffering = True


class _Backend:
    account = "acct"
    qos = "qos"
    partition = "part"
    mail_user = None
    requires_ray = False
    ray_dashboard_port = 8265
    ray_port = 6379
    endpoint = _EP()


class _Cfg:
    image = "/img/python.sif"
    wait_endpoint_s = 1800
    backend = _Backend()


def _render_lb():
    return _env().get_template("lb.sh.j2").render(
        cfg=_Cfg(),
        job_name="job",
        dep_jobid=111,
        replicas=2,
        swarm_directory="/swarm/s1",
        collector_script_path="/opt/watchdog_collector.py",
        supervisor_script_path="/opt/lb_supervisor.py",
    )


def test_lb_starts_supervisor_and_mounts_main_conf():
    out = _render_lb()
    assert "/opt/lb_supervisor.py" in out
    assert "lb_supervisor.py" in out
    assert "nginx.conf:/etc/nginx/nginx.conf:ro" in out
    # The old inline generator is gone:
    assert "generate_nginx_conf" not in out


def test_lb_still_starts_collector():
    out = _render_lb()
    assert "watchdog_collector.py" in out
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/backends/test_lb_template_render.py -k lb_ -v`
Expected: FAIL — `generate_nginx_conf` still present / supervisor not referenced.

- [ ] **Step 3: Rewrite `lb.sh.j2`**

Keep the sbatch header (lines 1–18) and the collector launch (lines 35–50) verbatim. Replace the whole nginx-generation/reload section with the supervisor. The driver must also render the main `nginx.conf` (add to Task 7's render block: `(serving_dir.parent / "nginx.conf").write_text(env.get_template("nginx.conf.j2").render())` → write to `$swarm_directory/nginx.conf`). New body after the collector block:

```bash
echo "[lb] waiting for $REPLICAS head files from job $DEP_JOBID"
t0=$(date +%s)
while true; do
    files=( $(ls "$SERVING_DIR/"*".head" 2>/dev/null || true) )
    {% raw %}
    if (( ${#files[@]} == REPLICAS )); then break; fi
    {% endraw %}
    if (( $(date +%s) - t0 > WAIT_SEC )); then
    {% raw %}
        echo "[lb] timeout: only ${#files[@]}/$REPLICAS replicas became healthy" >&2
    {% endraw %}
        exit 1
    fi
    sleep 5
done
echo "[lb] all replicas ready"

mkdir -p "$HOST_DIR/run"
PID_FILE="$HOST_DIR/run/nginx.pid"

# Initial upstreams config (one-shot) before nginx starts.
singularity exec --writable-tmpfs --bind "{{ supervisor_script_path }}:/opt/lb_supervisor.py,$HOST_DIR" "$PYTHON_IMG" \
  python3 /opt/lb_supervisor.py --once \
  --serving-dir "$SERVING_DIR" --pid-file "$PID_FILE" \
  {% if cfg.backend.requires_ray %}--ray-enabled --ray-dashboard-port {{ cfg.backend.ray_dashboard_port }} --ray-port {{ cfg.backend.ray_port }}{% endif %}

CACHE_DIR="$TMPDIR/cache"
mkdir -p "$CACHE_DIR"

singularity instance start --writable-tmpfs \
    -B "$HOST_DIR/nginx.conf:/etc/nginx/nginx.conf:ro" \
    -B "$SERVING_DIR:/etc/nginx/conf.d:ro" \
    -B "$HOST_DIR/run":/run \
    -B "$CACHE_DIR":/var/cache/nginx \
    -B "$HOST_DIR/logs/endpoint:/var/log/nginx" \
    "$NGINX_IMG" "$INSTANCE_NAME"

singularity exec instance://"$INSTANCE_NAME" bash -lc 'PATH=/usr/sbin:$PATH; nginx -t' \
  || { echo "[lb] ERROR: nginx config invalid" >&2; singularity instance stop "$INSTANCE_NAME" || true; exit 1; }
singularity exec instance://"$INSTANCE_NAME" bash -lc 'PATH=/usr/sbin:$PATH; nginx'
echo "[lb] started nginx instance '$INSTANCE_NAME'"

# Steady-state supervisor: keeps upstreams in sync and SIGHUPs nginx on change.
singularity exec --writable-tmpfs --bind "{{ supervisor_script_path }}:/opt/lb_supervisor.py,$HOST_DIR" "$PYTHON_IMG" \
  python3 /opt/lb_supervisor.py \
  --serving-dir "$SERVING_DIR" --pid-file "$PID_FILE" \
  {% if cfg.backend.requires_ray %}--ray-enabled --ray-dashboard-port {{ cfg.backend.ray_dashboard_port }} --ray-port {{ cfg.backend.ray_port }}{% endif %} \
  >>"$HOST_DIR/logs/supervisor.log" 2>&1 &
SUPERVISOR_PID=$!
echo "[lb] started supervisor PID $SUPERVISOR_PID"

trap 'echo "[lb] shutting down"; kill $COLLECTOR_PID $SUPERVISOR_PID 2>/dev/null; singularity instance stop "$INSTANCE_NAME" || true; exit 0' EXIT SIGTERM

wait $SUPERVISOR_PID
```

> Note: the image's built-in entrypoint is bypassed by running `nginx` explicitly; this matches the previous template's reliance on `nginx -s reload` against a manually started master. If the chosen nginx image needs `daemon off;`, the supervisor reload still works (SIGHUP), and the trap stops the instance.

- [ ] **Step 4: Update Task 7 render block to also write the main config**

Add to `submit_endpoint` (after writing `10-server.conf`):

```python
(Path(swarm_directory) / "nginx.conf").write_text(
    env.get_template("nginx.conf.j2").render()
)
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/backends/test_lb_template_render.py tests/backends/test_slurm_driver.py -v`
Expected: PASS.

- [ ] **Step 6: Full regression + lint + typecheck**

Run: `uv run pytest -m "not integration" -q && uv run ruff format . && uv run ruff check --fix . && uv run pyright`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "refactor(templates): drive lb.sh.j2 nginx config via lb_supervisor"
```

> **Phase 1 complete — this is a shippable, behavior-preserving milestone.** Validate on a real swarm before Phase 2 if possible (`domyn-swarm up`, confirm endpoint serves and reloads as replicas appear).

---

## Phase 2 — Monitoring feature

### Task 9: `MonitoringConfig` schema

**Files:**
- Modify: `src/domyn_swarm/config/slurm.py:27-39` (`SlurmEndpointConfig`)
- Test: `tests/config/test_monitoring_config.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/config/test_monitoring_config.py
from domyn_swarm.config.slurm import MonitoringConfig, SlurmEndpointConfig


def test_monitoring_disabled_by_default():
    ep = SlurmEndpointConfig()
    assert ep.monitoring.enabled is False


def test_monitoring_defaults():
    m = MonitoringConfig()
    assert m.mode == "container"
    assert m.port == 9090
    assert m.route_prefix == "/prometheus"
    assert m.scrape_interval == "15s"
    assert m.retention == "12h"
    assert m.exporter_port == 9113


def test_monitoring_route_prefix_normalized():
    assert MonitoringConfig(route_prefix="prometheus").route_prefix == "/prometheus"


def test_monitoring_mode_validated():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        MonitoringConfig(mode="rpm")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/config/test_monitoring_config.py -v`
Expected: FAIL — `ImportError: cannot import name 'MonitoringConfig'`.

- [ ] **Step 3: Implement**

```python
# add to src/domyn_swarm/config/slurm.py (above SlurmEndpointConfig)
class MonitoringConfig(BaseModel):
    """Optional Prometheus-based monitoring sidecar for the LB node.

    Disabled by default; when disabled the LB behaves exactly as before. See
    docs/superpowers/specs/2026-06-05-vllm-prometheus-monitoring-design.md.

    Attributes:
        enabled: Master switch. When False, all other fields are ignored.
        mode: 'container' (singularity images) or 'binary' (host binaries).
        prometheus_image: Singularity image for Prometheus (mode='container').
        nginx_exporter_image: Singularity image for nginx-prometheus-exporter.
        prometheus_binary: Prometheus binary name/path (mode='binary').
        nginx_exporter_binary: nginx-exporter binary name/path (mode='binary').
        port: Prometheus listen port on the LB node (proxied; not user-facing).
        exporter_port: nginx-exporter metrics port (scraped by Prometheus).
        route_prefix: nginx path prefix Prometheus is served under.
        scrape_interval: Prometheus global scrape interval (e.g. '15s').
        retention: TSDB retention window (e.g. '12h').
    """

    enabled: bool = False
    mode: Literal["container", "binary"] = "container"
    prometheus_image: str | None = Field(
        default_factory=default_for("slurm.endpoint.prometheus_image", None)
    )
    nginx_exporter_image: str | None = Field(
        default_factory=default_for("slurm.endpoint.nginx_exporter_image", None)
    )
    prometheus_binary: str = "prometheus"
    nginx_exporter_binary: str = "nginx-prometheus-exporter"
    port: int = 9090
    exporter_port: int = 9113
    route_prefix: str = "/prometheus"
    scrape_interval: str = "15s"
    retention: str = "12h"

    @field_validator("route_prefix")
    @classmethod
    def _ensure_leading_slash(cls, v: str) -> str:
        return v if v.startswith("/") else f"/{v}"
```

Then add to `SlurmEndpointConfig`:

```python
    monitoring: "MonitoringConfig" = Field(default_factory=lambda: MonitoringConfig())
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/config/test_monitoring_config.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(config): add MonitoringConfig to SlurmEndpointConfig"
```

### Task 10: Supervisor emits `targets.json`

**Files:**
- Modify: `src/domyn_swarm/runtime/lb_supervisor.py`
- Test: `tests/runtime/test_lb_supervisor.py`

- [ ] **Step 1: Write failing tests**

```python
# append to tests/runtime/test_lb_supervisor.py
import json


def test_render_targets_json(tmp_path: Path):
    (tmp_path / "replica-0.head").write_text("h0:9000")
    (tmp_path / "replica-1.head").write_text("h1:9000")
    payload = json.loads(lbs.render_targets(tmp_path))
    assert payload == [{"targets": ["h0:9000", "h1:9000"], "labels": {"job": "vllm"}}]


def test_render_targets_empty(tmp_path: Path):
    assert json.loads(lbs.render_targets(tmp_path)) == [{"targets": [], "labels": {"job": "vllm"}}]


def test_reconcile_writes_targets_when_enabled(tmp_path: Path, monkeypatch):
    serving = tmp_path / "serving"
    serving.mkdir()
    (serving / "replica-0.head").write_text("h0:9000")
    monkeypatch.setattr(lbs, "reload_nginx", lambda pid_file: True)
    opts = lbs.SupervisorOptions(
        serving_dir=serving, pid_file=tmp_path / "p", ray_enabled=False,
        ray_dashboard_port=8265, ray_port=6379, emit_targets=True,
    )
    lbs.reconcile_once(opts)
    assert (serving / "targets.json").exists()


def test_reconcile_no_targets_when_disabled(tmp_path: Path, monkeypatch):
    serving = tmp_path / "serving"
    serving.mkdir()
    (serving / "replica-0.head").write_text("h0:9000")
    monkeypatch.setattr(lbs, "reload_nginx", lambda pid_file: True)
    opts = lbs.SupervisorOptions(
        serving_dir=serving, pid_file=tmp_path / "p", ray_enabled=False,
        ray_dashboard_port=8265, ray_port=6379, emit_targets=False,
    )
    lbs.reconcile_once(opts)
    assert not (serving / "targets.json").exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/runtime/test_lb_supervisor.py -k targets -v`
Expected: FAIL — `render_targets` missing / `SupervisorOptions` has no `emit_targets`.

- [ ] **Step 3: Implement**

Add `import json` at the top. Add `emit_targets: bool = False` to `SupervisorOptions`. Add:

```python
TARGETS_FILENAME = "targets.json"


def render_targets(serving_dir: Path) -> str:
    """Render Prometheus file_sd targets JSON for the current replicas.

    Args:
        serving_dir: Directory containing ``replica-*.head`` files.

    Returns:
        A JSON string with a single file_sd entry labelling all replica
        ``host:port`` targets with ``job=vllm``.
    """
    addrs = read_head_files(serving_dir)
    return json.dumps([{"targets": addrs, "labels": {"job": "vllm"}}]) + "\n"
```

Update `reconcile_once` to also write targets when enabled (after the upstreams write):

```python
    if opts.emit_targets:
        write_if_changed(opts.serving_dir / TARGETS_FILENAME, render_targets(opts.serving_dir))
```

Add `--emit-targets` to `parse_args` (`action="store_true"`) and set `emit_targets=a.emit_targets` in the built `SupervisorOptions`. Add a matching field to the `parse_args` test if asserting it.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/runtime/test_lb_supervisor.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(runtime): emit Prometheus targets.json from lb_supervisor"
```

### Task 11: Gated nginx monitoring locations

**Files:**
- Modify: `src/domyn_swarm/templates/nginx_server.conf.j2`
- Test: `tests/backends/test_lb_template_render.py`

- [ ] **Step 1: Write failing tests**

```python
# append to tests/backends/test_lb_template_render.py
def _cfg_with_monitoring(enabled: bool):
    class Mon:
        pass
    mon = Mon()
    mon.enabled = enabled
    mon.port = 9090
    mon.route_prefix = "/prometheus"
    mon.exporter_port = 9113

    class EP:
        port = 9000
        nginx_timeout = "60s"
        enable_proxy_buffering = True
        monitoring = mon

    class Backend:
        requires_ray = False
        ray_dashboard_port = 8265
        ray_port = 6379
        endpoint = EP()

    class Cfg:
        backend = Backend()

    return Cfg()


def test_server_conf_adds_prometheus_locations_when_enabled():
    out = _env().get_template("nginx_server.conf.j2").render(cfg=_cfg_with_monitoring(True))
    assert "location /prometheus/" in out
    assert "proxy_pass http://127.0.0.1:9090" in out
    assert "location = /nginx_status" in out
    assert "stub_status;" in out
    assert "allow 127.0.0.1;" in out
    assert "deny all;" in out


def test_server_conf_no_prometheus_locations_when_disabled():
    out = _env().get_template("nginx_server.conf.j2").render(cfg=_cfg_with_monitoring(False))
    assert "/prometheus/" not in out
    assert "/nginx_status" not in out
```

> Update the earlier `nginx_server.conf.j2` tests (`test_server_conf_*`) to give their `EP` stand-in a `monitoring` attribute with `enabled = False`, since the template now references it.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/backends/test_lb_template_render.py -k prometheus -v`
Expected: FAIL — locations absent.

- [ ] **Step 3: Implement — add gated block inside the `server {}` in `nginx_server.conf.j2`** (before the closing `}`):

```jinja
  {% if cfg.backend.endpoint.monitoring.enabled %}
  location = /nginx_status {
    stub_status;
    allow 127.0.0.1;
    deny all;
  }
  location /prometheus/ {
    proxy_pass http://127.0.0.1:{{ cfg.backend.endpoint.monitoring.port }};
    proxy_set_header Host $host;
  }
  {% endif %}
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/backends/test_lb_template_render.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(templates): add gated nginx_status and prometheus proxy locations"
```

### Task 12: Render `prometheus.yml` + launch sidecars in `lb.sh.j2`

**Files:**
- Modify: `src/domyn_swarm/backends/serving/slurm_driver.py` (`submit_endpoint`)
- Create: `src/domyn_swarm/templates/prometheus.yml.j2`
- Modify: `src/domyn_swarm/templates/lb.sh.j2`
- Test: `tests/backends/test_lb_template_render.py`, `tests/backends/test_slurm_driver.py`

- [ ] **Step 1: Write failing tests**

```python
# append to tests/backends/test_lb_template_render.py
def test_prometheus_yml_render():
    class Mon:
        scrape_interval = "15s"
        exporter_port = 9113
        route_prefix = "/prometheus"

    out = _env().get_template("prometheus.yml.j2").render(monitoring=Mon())
    assert "scrape_interval: 15s" in out
    assert "job_name: vllm" in out
    assert "targets.json" in out
    assert "127.0.0.1:9113" in out


def test_lb_launches_sidecars_only_when_enabled():
    # _Cfg from Task 8, extended with monitoring enabled
    out = _render_lb_with_monitoring(enabled=True)
    assert "prometheus" in out
    assert "nginx-prometheus-exporter" in out or "nginx_exporter" in out
    assert "--web.route-prefix" in out
    out_off = _render_lb_with_monitoring(enabled=False)
    assert "--web.route-prefix" not in out_off
```

Add the helper near `_render_lb` (extends `_Backend.endpoint` with a `monitoring` object exposing `enabled`, `mode='container'`, `prometheus_image`, `nginx_exporter_image`, `prometheus_binary`, `nginx_exporter_binary`, `port`, `exporter_port`, `route_prefix`, `scrape_interval`, `retention`):

```python
def _render_lb_with_monitoring(enabled: bool):
    cfg = _Cfg()
    class Mon: ...
    m = Mon()
    m.enabled = enabled; m.mode = "container"
    m.prometheus_image = "/img/prom.sif"; m.nginx_exporter_image = "/img/nginxexp.sif"
    m.prometheus_binary = "prometheus"; m.nginx_exporter_binary = "nginx-prometheus-exporter"
    m.port = 9090; m.exporter_port = 9113; m.route_prefix = "/prometheus"
    m.scrape_interval = "15s"; m.retention = "12h"
    cfg.backend.endpoint.monitoring = m
    return _env().get_template("lb.sh.j2").render(
        cfg=cfg, job_name="job", dep_jobid=111, replicas=2,
        swarm_directory="/swarm/s1",
        collector_script_path="/opt/watchdog_collector.py",
        supervisor_script_path="/opt/lb_supervisor.py",
    )
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/backends/test_lb_template_render.py -k "prometheus_yml or sidecars" -v`
Expected: FAIL.

- [ ] **Step 3: Create `prometheus.yml.j2`**

```jinja
global:
  scrape_interval: {{ monitoring.scrape_interval }}

scrape_configs:
  - job_name: vllm
    metrics_path: /metrics
    file_sd_configs:
      - files:
          - /etc/prometheus/targets.json
  - job_name: nginx
    metrics_path: /metrics
    static_configs:
      - targets: ["127.0.0.1:{{ monitoring.exporter_port }}"]
```

- [ ] **Step 4: Render `prometheus.yml` in `submit_endpoint`** (only when enabled):

```python
mon = self.cfg.backend.endpoint.monitoring
if getattr(mon, "enabled", False):
    prom_yml = env.get_template("prometheus.yml.j2").render(monitoring=mon)
    (serving_dir / "prometheus.yml").write_text(prom_yml)
```

Pass `--emit-targets` to the supervisor invocations in `lb.sh.j2` when monitoring is enabled (add `{% if cfg.backend.endpoint.monitoring.enabled %}--emit-targets{% endif %}` to both supervisor calls).

- [ ] **Step 5: Add sidecar launch block to `lb.sh.j2`** (after nginx starts, before the steady-state supervisor):

```bash
{% if cfg.backend.endpoint.monitoring.enabled %}
MON_PROM_DIR="$TMPDIR/prometheus"
mkdir -p "$MON_PROM_DIR"
{% if cfg.backend.endpoint.monitoring.mode == "container" %}
singularity exec --writable-tmpfs \
  -B "$SERVING_DIR/prometheus.yml:/etc/prometheus/prometheus.yml:ro" \
  -B "$SERVING_DIR/targets.json:/etc/prometheus/targets.json:ro" \
  -B "$MON_PROM_DIR:/prometheus" \
  "{{ cfg.backend.endpoint.monitoring.prometheus_image }}" \
  prometheus --config.file=/etc/prometheus/prometheus.yml \
    --storage.tsdb.path=/prometheus \
    --storage.tsdb.retention.time={{ cfg.backend.endpoint.monitoring.retention }} \
    --web.route-prefix={{ cfg.backend.endpoint.monitoring.route_prefix }}/ \
    --web.listen-address=127.0.0.1:{{ cfg.backend.endpoint.monitoring.port }} \
  >>"$HOST_DIR/logs/prometheus.log" 2>&1 &
PROM_PID=$!
singularity exec --writable-tmpfs "{{ cfg.backend.endpoint.monitoring.nginx_exporter_image }}" \
  nginx-prometheus-exporter \
    --nginx.scrape-uri=http://127.0.0.1:{{ cfg.backend.endpoint.port }}/nginx_status \
    --web.listen-address=127.0.0.1:{{ cfg.backend.endpoint.monitoring.exporter_port }} \
  >>"$HOST_DIR/logs/nginx_exporter.log" 2>&1 &
NGINX_EXP_PID=$!
{% else %}
{{ cfg.backend.endpoint.monitoring.prometheus_binary }} \
  --config.file="$SERVING_DIR/prometheus.yml" \
  --storage.tsdb.path="$MON_PROM_DIR" \
  --storage.tsdb.retention.time={{ cfg.backend.endpoint.monitoring.retention }} \
  --web.route-prefix={{ cfg.backend.endpoint.monitoring.route_prefix }}/ \
  --web.listen-address=127.0.0.1:{{ cfg.backend.endpoint.monitoring.port }} \
  >>"$HOST_DIR/logs/prometheus.log" 2>&1 &
PROM_PID=$!
{{ cfg.backend.endpoint.monitoring.nginx_exporter_binary }} \
  --nginx.scrape-uri=http://127.0.0.1:{{ cfg.backend.endpoint.port }}/nginx_status \
  --web.listen-address=127.0.0.1:{{ cfg.backend.endpoint.monitoring.exporter_port }} \
  >>"$HOST_DIR/logs/nginx_exporter.log" 2>&1 &
NGINX_EXP_PID=$!
{% endif %}
echo "[lb] started monitoring sidecars (prometheus=$PROM_PID, nginx_exporter=$NGINX_EXP_PID)"
{% endif %}
```

Extend the EXIT/SIGTERM trap to also kill the sidecars when set:

```bash
trap 'echo "[lb] shutting down"; kill $COLLECTOR_PID $SUPERVISOR_PID ${PROM_PID:-} ${NGINX_EXP_PID:-} 2>/dev/null; singularity instance stop "$INSTANCE_NAME" || true; exit 0' EXIT SIGTERM
```

> The exporter scrapes nginx at `/nginx_status`, which is `allow 127.0.0.1; deny all;` — the exporter runs on the LB host so its requests originate from loopback.

- [ ] **Step 6: Run to verify pass**

Run: `uv run pytest tests/backends/test_lb_template_render.py tests/backends/test_slurm_driver.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat(slurm): launch prometheus + nginx-exporter sidecars on LB"
```

### Task 13: Bundled Grafana dashboard JSON

**Files:**
- Create: `src/domyn_swarm/data/dashboards/vllm.json`
- Create: `src/domyn_swarm/data/__init__.py`, `src/domyn_swarm/data/dashboards/__init__.py` (empty, so `importlib.resources` can locate the package)
- Test: `tests/data/test_dashboard.py`

- [ ] **Step 1: Write failing test**

```python
# tests/data/test_dashboard.py
import json
from importlib import resources

SUPPORTED = {"timeseries", "stat", "gauge", "bargauge", "table", "heatmap", "graph"}


def _load():
    with resources.files("domyn_swarm.data.dashboards").joinpath("vllm.json").open() as fh:
        return json.load(fh)


def test_dashboard_is_valid_json_with_title_and_panels():
    d = _load()
    assert d["title"]
    assert isinstance(d["panels"], list) and d["panels"]


def test_every_panel_has_supported_type_and_targets():
    d = _load()
    for p in d["panels"]:
        assert p["type"] in SUPPORTED, p
        assert p["title"]
        assert "gridPos" in p
        assert p["targets"] and p["targets"][0]["expr"]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/data/test_dashboard.py -v`
Expected: FAIL — resource missing.

- [ ] **Step 3: Create the dashboard** (a complete starting set; real vLLM/nginx metric names; correct panel types per the spec table)

```json
{
  "title": "domyn-swarm vLLM",
  "panels": [
    {
      "type": "timeseries",
      "title": "Generation throughput (tokens/s)",
      "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
      "targets": [
        {"expr": "sum(rate(vllm:generation_tokens_total[1m]))", "legendFormat": "tokens/s"}
      ]
    },
    {
      "type": "timeseries",
      "title": "Requests running / waiting",
      "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
      "targets": [
        {"expr": "sum(vllm:num_requests_running)", "legendFormat": "running"},
        {"expr": "sum(vllm:num_requests_waiting)", "legendFormat": "waiting"}
      ]
    },
    {
      "type": "timeseries",
      "title": "E2E request latency p95 (s)",
      "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
      "targets": [
        {"expr": "histogram_quantile(0.95, sum(rate(vllm:e2e_request_latency_seconds_bucket[5m])) by (le))", "legendFormat": "p95"}
      ]
    },
    {
      "type": "gauge",
      "title": "GPU KV-cache usage (%)",
      "gridPos": {"h": 8, "w": 6, "x": 12, "y": 8},
      "targets": [
        {"expr": "avg(vllm:gpu_cache_usage_perc) * 100", "legendFormat": "kv cache"}
      ]
    },
    {
      "type": "stat",
      "title": "Request success rate (1m)",
      "gridPos": {"h": 8, "w": 6, "x": 18, "y": 8},
      "targets": [
        {"expr": "sum(rate(vllm:request_success_total[1m]))", "legendFormat": "success/s"}
      ]
    },
    {
      "type": "timeseries",
      "title": "nginx active connections",
      "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16},
      "targets": [
        {"expr": "nginx_connections_active", "legendFormat": "active"}
      ]
    },
    {
      "type": "timeseries",
      "title": "nginx requests/s",
      "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16},
      "targets": [
        {"expr": "rate(nginx_http_requests_total[1m])", "legendFormat": "req/s"}
      ]
    }
  ]
}
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/data/test_dashboard.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Verify it ships in the wheel**

Run: `uv build && python -c "import zipfile,glob; w=glob.glob('dist/*.whl')[0]; print([n for n in zipfile.ZipFile(w).namelist() if 'dashboards' in n])"`
Expected: lists `domyn_swarm/data/dashboards/vllm.json`. (Hatchling includes package data automatically; no pyproject change needed.) Clean up: `rm -rf dist`.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat(data): add bundled vLLM/nginx Grafana dashboard"
```

### Task 14: `domyn-swarm monitor` command

**Files:**
- Create: `src/domyn_swarm/cli/monitor.py`
- Modify: `src/domyn_swarm/cli/main.py:106-111` (register command)
- Test: `tests/cli/test_monitor.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/cli/test_monitor.py
from types import SimpleNamespace
from unittest.mock import patch

from domyn_swarm.cli.monitor import build_prometheus_url, resolve_grafatui_argv


def _swarm(endpoint="http://lbnode:9000", enabled=True, route="/prometheus"):
    mon = SimpleNamespace(enabled=enabled, route_prefix=route)
    ep = SimpleNamespace(monitoring=mon)
    backend = SimpleNamespace(endpoint=ep)
    cfg = SimpleNamespace(backend=backend)
    return SimpleNamespace(endpoint=endpoint, cfg=cfg)


def test_build_prometheus_url_joins_route_prefix():
    assert build_prometheus_url(_swarm()) == "http://lbnode:9000/prometheus"


def test_build_prometheus_url_strips_trailing_slash():
    assert build_prometheus_url(_swarm(endpoint="http://lbnode:9000/")) == "http://lbnode:9000/prometheus"


def test_resolve_argv_includes_dashboard(tmp_path):
    dash = tmp_path / "vllm.json"
    dash.write_text("{}")
    argv = resolve_grafatui_argv("http://x/prometheus", dashboard=dash, extra=["--range", "1h"])
    assert argv[0] == "grafatui"
    assert "--prometheus-url" in argv and "http://x/prometheus" in argv
    assert "--grafana-json" in argv and str(dash) in argv
    assert "--range" in argv and "1h" in argv


def test_resolve_argv_without_dashboard():
    argv = resolve_grafatui_argv("http://x/prometheus", dashboard=None, extra=[])
    assert "--grafana-json" not in argv
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/cli/test_monitor.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `monitor.py`**

```python
# src/domyn_swarm/cli/monitor.py
# (license header as in other cli files)
"""``domyn-swarm monitor`` — launch grafatui against a swarm's Prometheus.

Lean by design: resolve the proxied Prometheus URL from persisted swarm state and
exec grafatui (an optional external tool). If grafatui is absent, print the URL and
an install hint so the user can use Grafana or run it manually.
"""

from __future__ import annotations

from importlib import resources
import os
from pathlib import Path
import shutil
from typing import Annotated

import typer


def build_prometheus_url(swarm) -> str:
    """Return the proxied Prometheus URL for a loaded swarm.

    Args:
        swarm: A loaded swarm object exposing ``endpoint`` and
            ``cfg.backend.endpoint.monitoring.route_prefix``.

    Returns:
        ``<endpoint><route_prefix>`` with duplicate slashes collapsed.
    """
    base = swarm.endpoint.rstrip("/")
    prefix = swarm.cfg.backend.endpoint.monitoring.route_prefix
    if not prefix.startswith("/"):
        prefix = "/" + prefix
    return f"{base}{prefix}"


def resolve_grafatui_argv(url: str, *, dashboard: Path | None, extra: list[str]) -> list[str]:
    """Build the grafatui argument vector.

    Args:
        url: Prometheus URL to pass via ``--prometheus-url``.
        dashboard: Optional Grafana dashboard JSON to load via ``--grafana-json``.
        extra: Additional passthrough arguments (e.g. ``--range 1h``).

    Returns:
        The full argv beginning with ``grafatui``.
    """
    argv = ["grafatui", "--prometheus-url", url]
    if dashboard is not None:
        argv += ["--grafana-json", str(dashboard)]
    argv += extra
    return argv


def _bundled_dashboard() -> Path | None:
    try:
        ref = resources.files("domyn_swarm.data.dashboards").joinpath("vllm.json")
        with resources.as_file(ref) as p:
            return Path(p)
    except (ModuleNotFoundError, FileNotFoundError):
        return None


def monitor(
    name: Annotated[str, typer.Argument(help="Swarm name to monitor.")],
    no_dashboard: Annotated[bool, typer.Option("--no-dashboard", help="Do not load the bundled dashboard.")] = False,
    prometheus_url: Annotated[str | None, typer.Option("--prometheus-url", help="Override the resolved Prometheus URL.")] = None,
    range_: Annotated[str | None, typer.Option("--range", help="grafatui time range, e.g. 1h.")] = None,
    step: Annotated[str | None, typer.Option("--step", help="grafatui query step, e.g. 15s.")] = None,
) -> None:
    """Launch grafatui pointed at the swarm's Prometheus endpoint."""
    from domyn_swarm.core.state.state_manager import SwarmStateManager

    swarm = SwarmStateManager.load(deployment_name=name)
    mon = swarm.cfg.backend.endpoint.monitoring
    if not getattr(mon, "enabled", False):
        typer.echo(
            f"Monitoring is not enabled for swarm '{name}'. "
            "Set backend.endpoint.monitoring.enabled: true and redeploy.",
            err=True,
        )
        raise typer.Exit(code=1)

    url = prometheus_url or build_prometheus_url(swarm)

    extra: list[str] = []
    if range_:
        extra += ["--range", range_]
    if step:
        extra += ["--step", step]

    if shutil.which("grafatui") is None:
        typer.echo(
            "grafatui not found on PATH. Prometheus is available at:\n"
            f"  {url}\n"
            "Install grafatui (`cargo install grafatui` or a GitHub release binary), "
            "or point Grafana at that URL.",
            err=True,
        )
        raise typer.Exit(code=127)

    dashboard = None if no_dashboard else _bundled_dashboard()
    argv = resolve_grafatui_argv(url, dashboard=dashboard, extra=extra)
    os.execvp(argv[0], argv)
```

- [ ] **Step 4: Register in `main.py`** (after the other `app.add_typer(...)` calls, around line 111):

```python
from .monitor import monitor as _monitor_cmd

app.command("monitor", short_help="Open grafatui against a swarm's Prometheus")(_monitor_cmd)
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/cli/test_monitor.py -v && uv run domyn-swarm monitor --help`
Expected: tests PASS; help shows the command.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat(cli): add 'domyn-swarm monitor' command"
```

### Task 15: Docs + example config + full verification

**Files:**
- Modify: `README.md` (monitoring section) and/or `examples/configs/` (add a commented `monitoring` block)
- Modify: `AGENTS.md` log layout (add `logs/prometheus.log`, `logs/nginx_exporter.log`, `logs/supervisor.log`, `serving/{10-server.conf,00-upstreams.conf,targets.json,prometheus.yml}`)

- [ ] **Step 1: Add an example monitoring block to `examples/configs/qwen3_32B.yaml`** (commented, disabled — minimal addition):

```yaml
  endpoint:
    port: 9001
    wall_time: "36:00:00"
    nginx_timeout: "8h"
    # monitoring:
    #   enabled: true
    #   prometheus_image: /path/to/prometheus.sif
    #   nginx_exporter_image: /path/to/nginx-prometheus-exporter.sif
    #   retention: 12h
```

- [ ] **Step 2: Update `AGENTS.md` log layout** to document the new serving-dir files and sidecar logs (see file structure list above), so the debugging map stays accurate.

- [ ] **Step 3: Add a README "Monitoring" subsection** documenting: enable via `monitoring.enabled`, what is scraped, the `/prometheus/` proxy URL, `domyn-swarm monitor <name>`, grafatui install, and the security note (the `/prometheus/` proxy is reachable by anyone who can reach the endpoint).

- [ ] **Step 4: Full verification gate**

Run:
```bash
uv run pytest -m "not integration" -q
uv run ruff format . && uv run ruff check .
uv run pyright
uv run pre-commit run --all-files
```
Expected: all pass. Investigate and fix any failure before claiming done (do not use `--no-verify`).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "docs: document Prometheus monitoring and monitor command"
```

---

## Self-review notes (author)

- **Spec coverage:** access via nginx proxy (Task 11, 14); vLLM+nginx scrape (Task 10, 12); node-local ephemeral TSDB (Task 12, `$TMPDIR` + retention); container-default/binary mode (Task 9, 12); lean `monitor` (Task 14); lb.sh.j2 leanness via Python supervisor (Tasks 1–8); bundled dashboard with correct panel types (Task 13); config minimal & default-off (Task 9). Aggregate-only nginx metrics accepted (stub_status in Task 11). All spec sections map to a task.
- **Type consistency:** `SupervisorOptions` fields (`serving_dir`, `pid_file`, `ray_enabled`, `ray_dashboard_port`, `ray_port`, `emit_targets`) are used consistently across Tasks 4/5/10; `reconcile_once`/`render_upstreams`/`render_targets`/`reload_nginx`/`write_if_changed` signatures match their call sites; `build_prometheus_url`/`resolve_grafatui_argv` match their tests.
- **Out of scope (unchanged from spec):** Ray multi-node target discovery refinement, node/GPU exporters, per-upstream nginx stats, Lepton, proxy auth, persisted TSDB.
- **Risk:** Task 0 (cross-container SIGHUP) gates the reload mechanism; fallback documented inline. If `nginx` started explicitly conflicts with the image entrypoint, the Phase 0 spike will reveal it.
```
