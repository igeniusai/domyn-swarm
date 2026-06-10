# vLLM / nginx Monitoring via a Prometheus LB Sidecar — Design

**Date:** 2026-06-05
**Status:** Approved (design); pending implementation plan
**Scope:** Slurm backend, standalone (non-Ray) vLLM replicas first

## Goal

Enable proper monitoring of vLLM metrics, replica health, and nginx load-balancing
for a Slurm-deployed swarm by running a **Prometheus sidecar on the load-balancer (LB)
node**. Prometheus ingests metrics from all vLLM replicas and from nginx, and is
consumable by either Grafana or a terminal-based viewer such as
[grafatui](https://github.com/fedexist/grafatui) (a TUI that connects directly to
Prometheus via `--prometheus-url`, no browser/tunnel required — well suited to HPC).

Configuration additions must be **minimal and additive**: existing configs and the
current behavior are byte-for-byte unchanged when monitoring is disabled (the default).

## Key decisions (from brainstorming)

| Decision | Choice |
| --- | --- |
| Access model | Proxy Prometheus through the existing nginx LB at `/prometheus/` on the user's existing endpoint URL. |
| Scrape scope | vLLM `/metrics` per replica **+** nginx aggregate metrics via an `nginx-prometheus-exporter` sidecar reading `stub_status`. No node/GPU exporters in v1. |
| TSDB lifecycle | Node-local `$TMPDIR`, ephemeral (dies with the LB job). Short configurable retention (default `12h`). Avoids small-fsync thrash on shared FS. |
| Convenience command | Ship a **lean** `domyn-swarm monitor <swarm>` that resolves the URL and execs grafatui, plus a bundled Grafana-JSON dashboard. grafatui stays an external optional tool. |
| Sidecar runtime | Default singularity **container** (mirrors existing nginx/python image pattern); `mode: binary` switches to configured binaries. |
| LB template | Refactor the big `lb.sh.j2` reload logic into a testable Python **supervisor** module; add monitoring on top. |

## Architecture

All new components run on the **existing LB node** (one Slurm job, `--exclusive`).
Replica nodes are untouched — vLLM already exposes `/metrics` on its serving port.

```
LB node (one Slurm job)
├── nginx          (existing) — + /nginx_status (local-only) + /prometheus/ proxy
├── collector      (existing) — unchanged (watchdog.db writer)
├── lb_supervisor  (NEW, Python in PYTHON_IMG) — watches replica-*.head →
│                    generates nginx.conf + targets.json → SIGHUP nginx
├── nginx-exporter (NEW, singularity/binary) — scrapes nginx stub_status → :9113/metrics
└── prometheus     (NEW, singularity/binary) — scrapes vLLM replicas + nginx-exporter;
                     TSDB in $TMPDIR (ephemeral); served behind /prometheus/
```

### Why piggyback target discovery on the existing `.head` files

The LB already regenerates `nginx.conf` from `serving/replica-*.head` files (each
containing `host:port`). Those are *exactly* the Prometheus scrape targets — vLLM
serves `/metrics` on the same port as the API. The supervisor generates `nginx.conf`
and `targets.json` from the **same `.head` list in the same pass**, so "what nginx
routes to" and "what Prometheus scrapes" can never drift. Prometheus picks up
`file_sd_configs` changes with no reload/signal.

## Component design

### LB supervisor (`src/domyn_swarm/runtime/lb_supervisor.py`)

A long-lived Python process (stdlib only), bind-mounted into and run under `PYTHON_IMG`
exactly like the collector. Responsibilities:

1. Watch `$SERVING_DIR/replica-*.head` (poll, same cadence as today, ~5s).
2. Generate `nginx.conf` (upstream block + server/locations) atomically (temp + `mv`).
   Reproduces the current template's nginx config generation, including the Ray
   upstreams block when `requires_ray` is set.
3. **(Phase 2)** Also emit `targets.json` for Prometheus `file_sd`:
   `[{"targets": ["host1:9001", ...], "labels": {"job": "vllm"}}]`.
4. Reload nginx via **SIGHUP to the nginx master PID** (see reload mechanism below).

This replaces the large bash `generate_nginx_conf` heredoc + reload loop in `lb.sh.j2`,
making that logic unit-testable Python instead of an untestable bash blob.

#### Reload mechanism (feasibility-sensitive — verify with a spike first)

A Python process inside `PYTHON_IMG` cannot invoke `singularity` (nested) and has no
`nginx` binary, so it cannot run `singularity exec instance://… nginx -s reload` /
`nginx -t` the way the current bash does. Resolution:

- Singularity instances do **not** create a new PID namespace by default, so the python
  container and the nginx container share the **host PID namespace**. The supervisor
  reloads nginx with `os.kill(master_pid, signal.SIGHUP)` (same uid → permitted).
- Expose the nginx master PID by writing the pidfile to the already-bind-mounted `/run`
  (`pid /run/nginx.pid;` in `nginx.conf.j2`) and binding `$HOST_DIR/run` into the
  supervisor container too.
- **Drop the explicit `nginx -t` pre-check.** SIGHUP reload validates the new config
  internally; on error it logs and keeps the running workers serving the old config, so
  safety is preserved without a separate validate step.

**Spike (do first in implementation):** confirm SIGHUP from the python container reloads
nginx running in the instance on this cluster's singularity build. Fallback if it does
not hold: keep a thin bash reload-trigger and let Python own only config generation.

### nginx template changes (`nginx.conf.j2` / generated config)

Only when monitoring is enabled:

- `location = /nginx_status { stub_status; allow 127.0.0.1; deny all; }` — local-only so
  the exporter can read it but remote callers (login node, users) cannot.
- `location /prometheus/ { proxy_pass http://127.0.0.1:<port>; }` — serves the Prometheus
  UI/API on the user's existing endpoint URL.

Always: `pid /run/nginx.pid;` (moved into bind-mounted `/run`) to support SIGHUP reload.

### Prometheus + nginx-exporter sidecars (`lb.sh.j2`, gated on `monitoring.enabled`)

- Write a static `prometheus.yml` into `$SERVING_DIR` once at startup with two scrape
  jobs: `vllm` (`file_sd_configs` → `targets.json`) and `nginx` (static
  `127.0.0.1:9113`). Run Prometheus with `--web.route-prefix=/prometheus/`,
  `--storage.tsdb.path=$TMPDIR/prometheus`,
  `--storage.tsdb.retention.time={{ retention }}`.
- Launch `nginx-prometheus-exporter` scraping `127.0.0.1:<nginx_port>/nginx_status`,
  exposing `:9113/metrics`.
- Both launched best-effort (`&`, PID captured); the existing `EXIT`/`SIGTERM` traps are
  extended to kill them. A sidecar failing to start must **not** take down the LB.

**Aggregate-only nginx metrics (accepted):** `stub_status` provides active connections,
accepts/handled, requests/s, reading/writing/waiting — not per-upstream distribution
(that needs the non-standard VTS module). Per-replica request/latency/throughput already
comes from each vLLM `/metrics`, so the combination shows load-balancing behavior.
Per-upstream nginx stats are a documented future iteration.

### `domyn-swarm monitor` command (`src/domyn_swarm/cli/monitor.py`)

Deliberately lean — one command, no new abstractions or dependencies:

- `domyn-swarm monitor <swarm-name>`: resolve LB host + endpoint port from the state DB
  (same lookup `status` uses); build `http://<lb-host>:<port>/prometheus`.
- `shutil.which("grafatui")` → if present, `exec` it with
  `--prometheus-url <url> --grafana-json <bundled dashboard>`; if absent, print the URL
  plus an install hint (`cargo install grafatui` / GitHub releases) so the user can fall
  back to Grafana or run it manually.
- Passthrough flags: `--range`, `--step`, `--no-dashboard`, `--prometheus-url` override.
- Bundled Grafana-JSON dashboard shipped as package data
  (`src/domyn_swarm/data/dashboards/vllm.json`), added to package-data in `pyproject.toml`.

#### Dashboard authoring (base + panel-type correctness)

**Base it on grafatui's own vLLM example**, then adapt metric names to our vLLM version:
`examples/demo/vllm_demo.json` and `examples/demo/vllm/grafana.json` in the grafatui repo
are existing vLLM dashboards — start from those rather than authoring from scratch.

grafatui supports seven panel types; the bundled dashboard must use the **correct `type`
per chart** (grafatui maps `type` → a specific TUI renderer):

| Metric / chart | Panel `type` | grafatui rendering |
| --- | --- | --- |
| Time-evolving series — tokens/s, TTFT, e2e latency, running/waiting requests, requests/s, nginx connections | `timeseries` | line chart (Braille) |
| Single current value — healthy replica count, total running requests | `stat` | big value + sparkline |
| Utilization ratio — KV-cache usage %, GPU cache usage % | `gauge` | horizontal gauge bar |
| Per-replica snapshot comparison | `bargauge` or `table` | vertical bars / two-col table |
| Latency-bucket distribution over time | `heatmap` | block heatmap |

grafatui compatibility constraints the dashboard JSON must respect:

- **Prometheus datasource only**; `targets[].datasource` is ignored — omit/leave generic.
- **All queries run as `query_range`** (no instant queries) — every panel works against a
  time range, which suits the panel choices above.
- Required fields per panel: `type`, `title`, `gridPos`, and a `targets` array with PromQL
  `expr`. `legendFormat` with `{{label}}` is supported (use it to label per-replica series,
  e.g. `{{instance}}`). Dashboard needs `title` + `panels`.
- Dashboard-level `uid`/`id`/`version`/`refresh`/`schemaVersion`/`timezone` are ignored —
  don't depend on them; keep the JSON minimal. Time range/step come from the `monitor`
  command's `--range`/`--step` flags, not the JSON.

The same JSON remains valid for real Grafana, so users wanting the browser UI can import
it unchanged.

## Configuration surface

One new optional model `MonitoringConfig` on `SlurmEndpointConfig` (in
`src/domyn_swarm/config/slurm.py`), default **disabled**. No state-DB change, no Alembic
migration (LB host is already persisted; `monitor` reads it).

```yaml
backend:
  endpoint:
    monitoring:
      enabled: false                 # master switch; all below ignored when false
      mode: container                # container | binary
      prometheus_image: ...          # default_for("slurm.endpoint.prometheus_image")
      nginx_exporter_image: ...      # default_for("slurm.endpoint.nginx_exporter_image")
      prometheus_binary: prometheus               # used when mode=binary
      nginx_exporter_binary: nginx-prometheus-exporter
      port: 9090                     # prometheus listen port (internal; proxied)
      route_prefix: /prometheus      # nginx path it is served under
      scrape_interval: 15s
      retention: 12h
```

Image defaults are wired through the defaults loader (`config/defaults.py`) mirroring
`nginx_image`; users supply their own `.sif`/binary as with the vLLM image.

## Error handling & security

- **Failure isolation:** monitoring sidecars are best-effort; failure logs and continues,
  never affecting nginx/collector. `monitoring.enabled=false` ⇒ zero behavioral change.
- **Security:** `/prometheus/` is exposed on the same port as the LLM API — anyone who can
  reach the endpoint can read metrics. Acceptable for internal HPC use; documented.
  Auth-on-the-proxy is a future option. `/nginx_status` is local-only.

## Testing

- `MonitoringConfig` defaults/validation; disabled-by-default.
- `lb_supervisor.py` unit tests: nginx.conf generation parity (incl. Ray block) and
  `targets.json` emission from a set of fake `.head` files.
- `lb.sh.j2` render snapshot: monitoring on vs off — sidecar launch blocks, trap
  extensions, nginx locations appear only when enabled; rendered-off output unchanged
  from current template.
- `monitor` command: mock state DB + `shutil.which` — grafatui present → correct argv;
  absent → prints URL + install hint.
- Bundled dashboard JSON: parses as JSON; every panel `type` is in grafatui's supported
  set (`timeseries`/`stat`/`gauge`/`bargauge`/`table`/`heatmap`/`graph`); every panel has
  `title`, `gridPos`, and a non-empty `targets` array with an `expr`.
- Phase-1 parity: behavior-preserving refactor verified before any monitoring is added.

## Implementation phases

1. **Phase 1 — behavior-preserving LB refactor.** Introduce `runtime/lb_supervisor.py`
   reproducing today's behavior (watch `.head` → generate `nginx.conf` → SIGHUP reload);
   `lb.sh.j2` shrinks to header + start collector + start nginx instance + start
   supervisor + traps; `pid /run/nginx.pid`. Snapshot/parity tests prove unchanged
   behavior. Includes the SIGHUP reload spike. No monitoring yet.
2. **Phase 2 — monitoring feature.** `MonitoringConfig`; supervisor emits `targets.json`;
   prometheus + nginx-exporter sidecars; `/nginx_status` + `/prometheus/` nginx locations;
   `domyn-swarm monitor` command + bundled dashboard.

Each phase is independently shippable and testable.

## Out of scope (v1)

- Ray-based multi-node replicas (standalone replicas first; Ray heads/workers later).
- node-exporter / dcgm-exporter (node + GPU metrics).
- Per-upstream nginx stats (VTS module).
- Lepton backend (platform-side log/metric capture differs).
- Auth on the `/prometheus/` proxy.
- Persisted/long-retention TSDB on shared FS.
