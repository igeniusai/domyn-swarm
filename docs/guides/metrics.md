# Metrics and dashboards

domyn-swarm can run a **Prometheus** instance next to the load balancer, scrape
every vLLM replica, and give you a dashboard over the result.

This is about *quantities* — tokens per second, queue depth, GPU utilisation. For
*is it broken*, see [Monitoring and troubleshooting](monitoring.md), which reads
health rather than metrics.

Off by default, and **Slurm only**: it is built out of sidecars on the
load-balancer node, which the Lepton backend does not have.

## Turning it on

```yaml
backend:
  type: slurm
  endpoint:
    monitoring:
      enabled: true
      prometheus_image: /path/to/prometheus.sif
      nginx_exporter_image: /path/to/nginx-prometheus-exporter.sif
      retention: 12h
```

`enabled` is a master switch: with it off, every other field here is ignored and
the load balancer behaves exactly as it did before monitoring existed.

Two images are needed because two sidecars run — Prometheus itself, and
`nginx-prometheus-exporter`, which turns Nginx's status page into metrics. To run
host binaries instead of containers:

```yaml
monitoring:
  enabled: true
  mode: binary
  prometheus_binary: /path/to/prometheus
  nginx_exporter_binary: /path/to/nginx-prometheus-exporter
```

Both binaries default to being looked up on `PATH` by name, so if they are
already installed cluster-wide, `mode: binary` alone is enough.

## What gets scraped

| Job | Target | Source |
| --- | --- | --- |
| `vllm` | every replica's `/metrics` | discovered from a file the supervisor keeps current |
| `nginx` | `nginx-prometheus-exporter` on `exporter_port` | static |
| `gpu` | each node's GPU exporter | only with `gpu_exporter.enabled` |
| `gpu_ownership` | `/gpu_ownership` on the endpoint | only with `gpu_exporter.enabled` |
| `ray` | each node's Ray metrics port | only with `ray_metrics.enabled` |

Replica targets are not static — replicas come and go, and their host and port are
only known once Slurm has placed them. Each replica writes a `replica-<id>.head`
file into the swarm's serving directory, and the load-balancer supervisor turns
those into both Nginx's upstreams and Prometheus's target file. Adding a replica
therefore adds a scrape target with no reconfiguration.

Every series is labelled `swarm` with the swarm's name, from Prometheus's
`external_labels`.

## Reaching it

Prometheus is served **through the load balancer**, at `route_prefix` on the same
endpoint URL as the model:

```
http://<endpoint>/prometheus
```

The `port` field (default `9090`) is where Prometheus listens on the
load-balancer node, behind that proxy — it is not the port you connect to.

:::{warning}
Nothing authenticates `/prometheus`. Anyone who can reach the swarm's endpoint
can read it, and can read the metrics of every replica. Treat it as internal to
your cluster.
:::

The database is **node-local and ephemeral**. It lives on the load-balancer node
and disappears when that job ends, so `retention` (default `12h`) only caps a
single run's history. If you need metrics to outlive a swarm, point an external
Prometheus at the same `/prometheus` URL and let it federate.

## Viewing a dashboard

```bash
domyn-swarm monitor my-swarm-name
```

This resolves the swarm's Prometheus URL from its state record and hands it to
`grafatui`, a terminal Grafana dashboard renderer, along with the bundled vLLM
dashboard. It replaces the current process, so you get grafatui's UI directly.
Install it with `cargo install grafatui` or from a GitHub release binary.

grafatui is not a dependency of domyn-swarm and is not installed with it. Without
it on `PATH` the command exits **127** and prints the Prometheus URL, so you can
point a real Grafana at it instead.

Other exits worth knowing, because they are configuration problems rather than
failures:

| Exit | Meaning |
| --- | --- |
| `1` | monitoring is not enabled for this swarm — enable it and redeploy |
| `2` | `--gpu` was passed but no GPU exporter is configured, or `--dashboard` names a file that does not exist |
| `127` | `grafatui` is not on `PATH` |

Note that "enable it and redeploy" is the whole story for exit 1: monitoring is
wired into the load-balancer job at submission time, so it cannot be switched on
under a swarm that is already up.

### Useful flags

```bash
domyn-swarm monitor my-swarm --range 1h --step 15s
domyn-swarm monitor my-swarm -d ./my-dashboard.json
domyn-swarm monitor my-swarm --prometheus-url http://other-host:9090
domyn-swarm monitor my-swarm --var replicas=8
```

`--range` and `--step` pass straight through to grafatui. `--prometheus-url`
overrides the resolved URL, which is how you point at a swarm's Prometheus from
outside the cluster or through a tunnel.

### Dashboard variables

The bundled dashboard is parameterised, and `monitor` fills two variables from the
swarm config: `vllm_job` (the Prometheus job name, `vllm`) and `replicas`, which
drives the *Replicas down* panel. Override either, or add variables your own
dashboard needs, with repeatable `--var KEY=VALUE`.

Two variables you might expect are deliberately **not** filled in:

- `swarm` — each swarm runs its own Prometheus scraping only itself, so filtering
  by swarm would be redundant.
- `model` — vLLM labels metrics with the full resolved model path, not the `model`
  string from your config, so an auto-filled value would match nothing. Pass it
  explicitly with `--var` if your dashboard uses it.

## GPU metrics

Per-node GPU metrics come from a separate exporter, enabled underneath
`monitoring`:

```yaml
backend:
  endpoint:
    monitoring:
      enabled: true
      gpu_exporter:
        enabled: true
        kind: nvidia_smi
        image: /shared/images/gpu_exporter_nvidia_smi.sif
```

Then:

```bash
domyn-swarm monitor my-swarm-name --gpu
```

which loads the bundled dashboard matching the configured `kind`.

### Choosing a kind

`nvidia_smi` (the default)
: A small static binary that shells out to `nvidia-smi`. Portable, runs
  unprivileged anywhere `nvidia-smi` exists, and is the right default.

`dcgm`
: NVIDIA's Data Center GPU Manager, emitting the standard `DCGM_FI_*` series.
  Pinned to the **3.x** line because 4.x aborts when run unprivileged.
  Driver-to-DCGM compatibility varies by site, so this is the one to try when you
  need the standard metric names and have checked your driver supports it.

Either way the exporter runs unprivileged inside the job's GPU cgroup, and covers
memory, utilisation, power, temperature, and clocks with throttle reasons.
Profiling counters (`DCGM_FI_PROF_*`) need root and are not available.

### Images and modes

Build from the recipes shipped in `images/`:

```bash
sudo singularity build gpu_exporter_nvidia_smi.sif images/gpu_exporter_nvidia_smi.def
sudo singularity build gpu_exporter_dcgm.sif images/gpu_exporter_dcgm.def
```

The exporter follows `monitoring.mode`, and the two kinds do not support the same
modes. The config rejects the impossible combinations at load time rather than
failing on the node:

- `nvidia_smi` with `mode: container` needs an explicit `gpu_exporter.image` —
  there is no default image to fall back to. Omitting it raises *nvidia_smi
  container mode needs an explicit gpu_exporter.image*.
- `dcgm` only works with `mode: container`, because it is launched through
  `singularity exec`. Asking for `mode: binary` raises *dcgm exporter is only
  supported in container mode*.
- `dcgm` with `mode: container` and no image falls back to a public NVIDIA image.

So `mode: binary` means `kind: nvidia_smi`, with `binary` defaulting to
`nvidia_gpu_exporter` on `PATH`.

### Which replica owns which GPU

An exporter runs once per *node*, but a node can host several replicas. To
attribute a GPU to a replica, each replica records the UUIDs of the GPUs it owns,
and the supervisor renders them as a join metric:

```
dswarm_gpu_owner{uuid="GPU-...", UUID="GPU-...", replica="0"} 1
```

Both spellings of the label are emitted on purpose: the `nvidia_smi` exporter
labels GPUs `uuid` and DCGM labels them `UUID`, so one series joins against
either without the dashboard caring which exporter is running.

Prometheus scrapes this from `/gpu_ownership` on the endpoint, served by the same
Nginx that fronts the model.

## Ray metrics

For multi-node replicas, Ray's own `ray_*` metrics are scraped from every node.
This needs no configuration: `ray_metrics.enabled` resolves itself to true when
monitoring is on and the deployment requires Ray, and to false otherwise.

Set it explicitly to `false` to opt out, or `true` to force it on. An explicit
value is always respected, and after validation the field is never left unset.

```yaml
monitoring:
  enabled: true
  ray_metrics:
    enabled: false     # scrape vLLM but not Ray
```

When Ray metrics are active, `domyn-swarm monitor` appends a group of Ray panels
to the bundled dashboard rather than using a separate one, so a Ray swarm's
dashboard is the vLLM dashboard plus cluster panels. Passing `--dashboard`
suppresses this — your dashboard is used exactly as given.

`ray_metrics.port` (default `8090`) is Ray's `--metrics-export-port`. It is fixed
rather than ephemeral so that the per-node files Prometheus discovers have stable
contents.

## What monitoring does not do

- **It does not replace health checks.** Prometheus tells you a replica is slow;
  the watchdog decides whether it is dead. See
  [Watchdog and collector](../concepts/watchdog-collector.md).
- **It does not persist.** No metric outlives the load-balancer job unless
  something external is federating.
- **It does not alert.** No Alertmanager is deployed and no rules are shipped.

## Full field reference

Every field, with types and defaults, is generated from the models:
`MonitoringConfig`, `GpuExporterConfig` and `RayMetricsConfig` in
[Configuration](../reference/configuration.md).
