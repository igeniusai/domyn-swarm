# grafatui — feature requests & bug reports (for upstream)

Prepared for filing against https://github.com/fedexist/grafatui.

These come from integrating grafatui as the default terminal monitoring UI for
**domyn-swarm**: vLLM serving endpoints on Slurm/HPC, with Prometheus running as a
sidecar on the load-balancer node and exposed behind an nginx subpath. grafatui is run
from a login node and pointed at that Prometheus.

## How to use this document

- Each section below is a **self-contained, copy-pasteable GitHub issue** (title +
  body). File the ones you want as separate issues.
- Two items (**#1 macros**, **#7 subpath**) are marked **VERIFY FIRST** — please confirm
  against the current binary before filing; they may already work, in which case skip or
  convert to a docs question.
- Suggested filing order by value: **#1, #2** first (highest impact), then #3, #5,
  #6, #8 as capacity allows.

### Checked against grafatui v0.1.8 (`grafatui --help`)

The real CLI was inspected, which updates this list:

- **#4 (auto-refresh) — ALREADY SUPPORTED, do not file.** `--refresh-rate <MS>` (default
  1000 ms) is the data refresh interval; `--tick-rate` controls redraw. Kept below only as
  a struck-through record.
- **#2 (variables) — partially present.** `--var KEY=VALUE` exists for *overriding* a
  variable's value; the request remaining is *query-based population* (`label_values(...)`)
  and an *interactive in-TUI selector*.
- **#3 (auth) — confirmed ABSENT in v0.1.8** (no `--bearer-token`/`--header`/`--insecure`);
  request stands.
- **#8 (env vars) — `--config <FILE>` (TOML) exists**, but no environment-variable support;
  the env-var part of the request stands.
- Also available and useful for us: `--query <EXPR>` (append ad-hoc PromQL panels),
  `--theme`, `--export-dir`/`--export-format` (svg/png), `--var`, `--config`.

## Shared context (optional — paste into any issue)

> **Use case:** terminal-based monitoring of vLLM inference endpoints on an HPC/Slurm
> cluster (no browser/tunneling). Prometheus runs as a sidecar on the load-balancer node
> and is reverse-proxied by nginx under a path prefix (`http://<host>:<port>/prometheus/`).
> grafatui runs on a login node. Several vLLM replicas are scraped, each labeled by
> `instance` (host:port). We want to reuse existing vLLM Grafana dashboards where possible.

---

## 1. [VERIFY FIRST] Support Grafana time macros (`$__rate_interval`, `$__interval`, `$__range`)

**Title:** Expand Grafana time macros (`$__rate_interval`, `$__interval`, `$__range`) in panel queries

**Body:**

Most real-world Grafana dashboards — including the official vLLM dashboard — write rate
queries using Grafana's time macros, e.g.:

```promql
rate(vllm:request_success_total[$__rate_interval])
sum(rate(vllm:prompt_tokens_total[$__rate_interval]))
```

If grafatui does not expand `$__rate_interval` / `$__interval` / `$__range` before sending
the query to Prometheus, these panels fail or return nothing, and "import existing
dashboard JSON" effectively doesn't work for production dashboards (the user must
hand-edit every `expr` to hardcode an interval).

**Request:** expand the common Grafana time macros against the active query range/step
(`$__rate_interval`, `$__interval`, `$__range`, ideally also `$__interval_ms` and
`$__rate_interval_ms`).

**Acceptance criteria:**
- A panel with `rate(metric_total[$__rate_interval])` renders correctly without editing.
- The official vLLM Grafana dashboard imports and renders its rate-based panels.

*(Please confirm current behavior first — if `examples/demo/vllm_demo.json` hardcodes
intervals rather than using macros, that suggests macros aren't supported yet.)*

---

## 2. Query-based dashboard variables (`label_values()`) with interactive TUI selection

**Title:** Support query-based template variables and interactive selection in the TUI

**Body:**

When monitoring multiple targets (e.g. several vLLM replicas, each labeled by `instance`),
a dashboard variable that populates from Prometheus and can be selected interactively is
very valuable:

```text
variable: $instance
query:    label_values(vllm:num_requests_running, instance)
```

Today `--var` can override a variable's value, but there is no query-based population
(`label_values(...)`) and no in-TUI selector to switch between values.

**Request:**
- Resolve query-based variables via `label_values(...)` / `query_result(...)`.
- A TUI control (dropdown/list, keybind) to switch the selected value and re-render
  panels whose `expr` references `$variable`.
- Support `All` / multi-select where feasible.

**Acceptance criteria:**
- A dashboard with an `instance` variable shows a selector listing discovered replicas.
- Selecting a replica filters all panels using `{instance="$instance"}`.

---

## 3. Authentication and TLS options (bearer token, basic auth, custom header, insecure)

**Title:** Support auth headers / basic auth / `--insecure` for the Prometheus connection

**Body:**

When Prometheus is exposed through a reverse proxy, access is often protected (bearer
token, basic auth) and/or served over HTTPS with an internal/self-signed cert. grafatui
currently appears to assume an unauthenticated plain-HTTP endpoint.

**Request (any/all):**
- `--bearer-token <token>` (and/or `--header 'Authorization: Bearer ...'`, repeatable).
- Basic auth via `--prometheus-url https://user:pass@host` or `--username/--password`.
- `--insecure` to skip TLS verification for internal certs.
- Read these from env vars (e.g. `GRAFATUI_BEARER_TOKEN`) for clean scripted launches.

**Acceptance criteria:**
- grafatui can query a token-protected, HTTPS Prometheus behind a proxy.

---

## 4. ~~Auto-refresh / live mode~~ — ALREADY SUPPORTED (do not file)

**Resolved against v0.1.8.** grafatui already refreshes data on an interval via
`--refresh-rate <MS>` (default 1000 ms; `--tick-rate` controls UI redraw). No request
needed. The only possible follow-up would be honoring a dashboard JSON `refresh` field as
an alternative to the flag — low value given `--refresh-rate` exists.

*(Please confirm whether a manual-refresh keybind already exists — if so, this is just the
automatic-interval addition.)*

---

## 5. Instant queries for `stat` / `gauge` / `bargauge` / `table` panels

**Title:** Use instant queries (`/api/v1/query`) for single-value panel types

**Body:**

Per the compatibility docs, all panels use `query_range`. For panels that display a single
current value (`stat`, `gauge`, `bargauge`, `table`), an instant query (`/api/v1/query`) is
both more correct (true "now" value rather than the last sample of a range) and cheaper.

**Request:** issue instant queries for single-value panel types (or expose a per-panel /
global toggle), keeping `query_range` for time-series panels.

**Acceptance criteria:**
- A `stat`/`gauge` panel reflects the instantaneous value, matching Grafana's behavior.

---

## 6. First-class histogram → `heatmap` support

**Title:** Render Prometheus histogram (`_bucket` / `le`) series as the heatmap panel

**Body:**

vLLM exposes latency as Prometheus histograms, e.g.
`vllm:e2e_request_latency_seconds_bucket` and `vllm:time_to_first_token_seconds_bucket`
with `le` buckets. Native support for turning classic histogram buckets into the `heatmap`
panel (handling cumulative `le` buckets correctly) would let us visualize latency
distributions properly.

**Acceptance criteria:**
- A `heatmap` panel querying `*_bucket` series renders a latency-over-time heatmap with
  correct bucket handling.

---

## 7. [VERIFY FIRST] Robust base-URL / path-prefix handling

**Title:** Honor a path prefix in `--prometheus-url` (don't strip the base path)

**Body:**

We expose Prometheus behind a reverse proxy under a path prefix and run Prometheus with
`--web.route-prefix=/prometheus/`, so the base URL is:

```text
http://<host>:<port>/prometheus
```

API calls must be appended to the full path, e.g.
`http://<host>:<port>/prometheus/api/v1/query_range`. grafatui must not strip or ignore the
`/prometheus` prefix.

**Acceptance criteria:**
- `grafatui --prometheus-url http://host:port/prometheus` queries
  `…/prometheus/api/v1/query_range` and renders data.

*(Likely already works — please verify; if it does not, this is a focused bug report.)*

---

## 8. Read connection settings from environment variables

**Title:** Allow `--prometheus-url` (and key flags) to be set via environment variables

**Body:**

For scripted launches (we wrap grafatui in a `domyn-swarm monitor` command), reading the
Prometheus URL and common options from env vars (e.g. `GRAFATUI_PROMETHEUS_URL`,
`GRAFATUI_RANGE`, `GRAFATUI_STEP`) avoids threading CLI args and complements the existing
`--config` TOML.

**Acceptance criteria:**
- With `GRAFATUI_PROMETHEUS_URL` set, `grafatui` (no `--prometheus-url`) connects to it;
  an explicit flag still overrides the env var.
