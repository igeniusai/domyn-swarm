<p align="center">
  <picture>
      <source srcset="static/domyn-swarm-logo-white.svg" media="(prefers-color-scheme: dark)">
      <source srcset="static/domyn-swarm-logo-primary.svg" media="(prefers-color-scheme: light)">
      <img src="static/domyn-swarm-logo-primary.svg" alt="domyn-swarm" height="100">
   </picture>
</p>

> A simple, batteries‑included CLI and Python library for launching **LLM serving endpoints** and running **high‑throughput batch jobs** against them. First‑class support for **Slurm** (HPC) and **NVIDIA DGX Cloud Lepton**.

<p align="center">
<a href="https://igeniusai.github.io/domyn-swarm/"><img src="https://img.shields.io/badge/docs-latest-blue" alt="Documentation"></a>
<img src="https://github.com/igeniusai/domyn-swarm/actions/workflows/ci.yaml/badge.svg" alt="CI">
<img src="https://img.shields.io/badge/python-3.10%7C3.11%7C3.12%7C3.13-brightgreen?style=flat&logoColor=green" alt="Python">
<img src="https://img.shields.io/badge/License-Apache%202.0-blue" alt="License - Apache 2.0">
<img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff">
<img src="https://microsoft.github.io/pyright/img/pyright_badge.svg" alt="Pyright">
</p>

---

## Why Domyn‑Swarm?

Domyn‑Swarm gives you a **single, consistent workflow** to stand up a **scalable
LLM endpoint** (vLLM, OpenAI‑compatible), **submit jobs and scripts** that call it
with **checkpointing, retries and bounded concurrency**, and **tear it all down**
cleanly — the same way on HPC (Slurm) and in the cloud (Lepton).

It's designed for **fast evaluation loops**, **robust batch inference**, and
**easy backend extension**.

---

## Installation

**PyPI (once published):**

> [!NOTE]
> We still haven't published the package on PyPI, but it will soon be available

```bash
pip install domyn-swarm
# Optional extras
pip install 'domyn-swarm[lepton]'   # NVIDIA DGX Cloud Lepton backend
pip install 'domyn-swarm[polars]'   # Polars data backend
pip install 'domyn-swarm[ray]'      # Ray Data backend
# or everything
pip install 'domyn-swarm[all]'
```

**From source (GitHub):**

```bash
RELEASE=v0.29.0
uv tool install --from git+ssh://git@github.com/igeniusai/domyn-swarm.git@$RELEASE --python 3.12 domyn-swarm
```

Full instructions, including `uv add`, editable installs and the Lepton
`lep login` step, are in
[Installation](https://igeniusai.github.io/domyn-swarm/latest/getting-started/installation.html).

---

## Quickstart

```yaml
# config.yaml
model: "HuggingFaceTB/SmolLM3-3B-Base"
gpus_per_replica: 16
replicas: 2
backend:
  type: slurm
  partition: partition_name    # your HPC partition
  account: account_name        # your HPC account
  qos: qos_name                # qos for cluster + jobs
```

```bash
# Launch replicas behind a load balancer, and wait until they answer
domyn-swarm up -c config.yaml

# Run a batch job against the endpoint: DataFrame in, DataFrame out
domyn-swarm job submit \
  --name my-swarm-name \
  --input examples/data/chat_completion.parquet \
  --output results.parquet

# Check on it
domyn-swarm status my-swarm-name

# Remove everything it created
domyn-swarm down my-swarm-name
```

The [Quickstart](https://igeniusai.github.io/domyn-swarm/latest/getting-started/quickstart.html)
walks through the same run with an explanation of each step.

---

## Documentation

The full documentation lives at
**[igeniusai.github.io/domyn-swarm](https://igeniusai.github.io/domyn-swarm/)**.

* [Getting started](https://igeniusai.github.io/domyn-swarm/latest/getting-started/index.html)
  — install, launch a swarm, write your first custom job
* [Guides](https://igeniusai.github.io/domyn-swarm/latest/guides/index.html)
  — Slurm and Lepton, submitting jobs, checkpointing, sharding, data backends,
  swarm state, monitoring
* [Concepts](https://igeniusai.github.io/domyn-swarm/latest/concepts/index.html)
  — architecture, the backend protocols, the `SwarmJob` lifecycle, configuration
  precedence
* Reference, generated from the source on every build —
  [CLI](https://igeniusai.github.io/domyn-swarm/latest/reference/cli.html),
  [Configuration](https://igeniusai.github.io/domyn-swarm/latest/reference/configuration.html),
  [Environment variables](https://igeniusai.github.io/domyn-swarm/latest/reference/environment.html),
  [Python API](https://igeniusai.github.io/domyn-swarm/latest/reference/api/index.html)

---

## Monitoring (optional)

Domyn-Swarm can run a **Prometheus** instance plus an **nginx-prometheus-exporter** as sidecars
on the LB node to collect vLLM and load-balancer metrics. This is **off by default** and currently
**Slurm only**. Enable it under `backend.endpoint.monitoring`:

```yaml
backend:
  type: slurm
  endpoint:
    monitoring:
      enabled: true
      prometheus_image: /path/to/prometheus.sif                       # singularity (default mode)
      nginx_exporter_image: /path/to/nginx-prometheus-exporter.sif
      retention: 12h
      # mode: binary                                                  # or run host binaries instead
      # prometheus_binary: /path/to/prometheus
      # nginx_exporter_binary: /path/to/nginx-prometheus-exporter
```

**What is scraped**: each vLLM replica's `/metrics` endpoint, plus aggregate nginx metrics exposed
by the nginx-prometheus-exporter. Targets are kept in sync automatically by the LB supervisor.

**Where it lives**: Prometheus is reverse-proxied by the LB at `http://<endpoint>/prometheus`
(the same endpoint URL as the swarm). The TSDB is **node-local and ephemeral** — it disappears when
the LB job ends, and retention defaults to `12h`.

**View it** with the bundled vLLM/nginx dashboard:

```bash
domyn-swarm monitor <swarm-name>
```

This launches `grafatui` pointed at `http://<endpoint>/prometheus`. Install it via
`cargo install grafatui` or a GitHub release binary. Useful flags: `--dashboard/-d`
(load a custom Grafana dashboard JSON instead of the bundled one), `--range`, `--step`,
`--prometheus-url`. Alternatively, point a regular Grafana instance at the same
`/prometheus` URL.

The bundled dashboard is parameterized by template variables that `monitor` auto-fills
from the swarm config: `vllm_job` (the Prometheus job name) and `replicas` (the expected
replica count, which drives the dynamic "Replicas down" panel). Override them — or pass
extra variables your own dashboard uses — with repeatable `--var KEY=VALUE`, e.g.
`domyn-swarm monitor <swarm> --var replicas=8`. Each swarm runs its own Prometheus that
scrapes only that swarm, so the dashboard does not filter by swarm or model.

> **Security note**: the `/prometheus/` path is reachable by anyone who can reach the endpoint. There is
> no authentication on it, so this feature is intended for internal/HPC use only.

### GPU monitoring

Optional per-node GPU monitoring is available via configurable **NVIDIA GPU exporters** (nvidia-smi or DCGM).
Enable it under `backend.endpoint.monitoring.gpu_exporter`:

> GPU monitoring currently covers non-Ray (single-node-per-replica) deployments; Ray (multi-node) support is a planned follow-up.

```yaml
backend:
  endpoint:
    monitoring:
      enabled: true
      gpu_exporter:
        enabled: true
        kind: nvidia_smi           # default: portable, unprivileged; or 'dcgm' (standard metrics)
        port: 9835                 # optional, defaults to 9835
        # image: /path/to/image.sif    # required only for nvidia_smi + mode=container
        # binary: /path/to/binary      # required only for dcgm + mode=binary
```

**Exporter kinds:**

- `nvidia_smi` (default) — lightweight, portable. A 12MB static binary that runs unprivileged on any site with `nvidia-smi` available. Uses basic GPU metrics.
- `dcgm` — standard NVIDIA Data Center GPU Manager metrics (`DCGM_FI_*` series). Pinned to the **3.x line** because 4.x aborts when running unprivileged; driver↔DCGM compatibility varies by site.

**Building and running:**

Build exporter images from the shipped recipes in `images/`:

```bash
sudo singularity build gpu_exporter_nvidia_smi.sif images/gpu_exporter_nvidia_smi.def
sudo singularity build gpu_exporter_dcgm.sif images/gpu_exporter_dcgm.def
```

Then reference them in your config:

```yaml
gpu_exporter:
  kind: nvidia_smi
  image: /shared/images/gpu_exporter_nvidia_smi.sif
```

Alternatively, run with `mode: binary` (set at `backend.endpoint.monitoring.mode`). Note that
`dcgm` is only supported in `mode: container` (it runs via `singularity exec`); `mode: binary`
only supports `kind: nvidia_smi`:

```yaml
monitoring:
  mode: binary
  gpu_exporter:
    kind: nvidia_smi
    binary: /usr/local/bin/nvidia_gpu_exporter
```

**Accessing the GPU dashboard:**

View real-time GPU metrics and trends using:

```bash
domyn-swarm monitor <swarm-name> --gpu
```

This launches `grafatui` pointed at the configured GPU exporter, using the bundled dashboard parameterized by `kind` (nvidia_smi or dcgm). For additional options, see `domyn-swarm monitor --help`.

**Metrics available:**

GPU exporters run **unprivileged** inside the job's GPU cgroup. The following metrics are available:

- Memory (allocated, free, used)
- Utilization (compute, memory)
- Power usage and limit
- Temperature
- Clock speeds and throttle status

> **Note**: Profiling metrics (`DCGM_FI_PROF_*` for DCGM) require root and are not available in unprivileged mode.

**GPU attribution:**

Each GPU is mapped to its replica via a **UUID → replica join**. The exporter runs once per compute node and emits per-GPU and per-replica metrics, allowing you to correlate performance with specific model instances across your cluster.

---

## Contributing

We welcome issues and PRs! Please see:

* `CONTRIBUTING.md` — how to propose changes, coding style, DCO/CLA (as applicable)

---

## License

Licensed under the **Apache License, Version 2.0**. See `LICENSE` and `NOTICE`.

---

## Acknowledgements

* Built on **vLLM** and **Ray** (Apache‑2.0)
* Optional **NVIDIA DGX Cloud Lepton** integration via the Lepton SDK

Happy swarming! 🚀
