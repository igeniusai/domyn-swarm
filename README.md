<p align="center">
  <picture>
      <source srcset="static/domyn-swarm-logo-white.svg" media="(prefers-color-scheme: dark)">
      <source srcset="static/domyn-swarm-logo-primary.svg" media="(prefers-color-scheme: light)">
      <img src="static/domyn-swarm-logo-primary.svg" alt="domyn-swarm" height="100">
   </picture>
</p>

> A simple, batteries‑included CLI and Python library for launching **LLM serving endpoints** and running **high‑throughput batch jobs** against them. First‑class support for **Slurm** (HPC) and **NVIDIA DGX Cloud Lepton**.

<p align="center">
<a href="https://domynswarm.domyn.com/"><img src="https://img.shields.io/badge/docs-latest-blue" alt="Documentation"></a>
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
RELEASE=v0.30.0
uv tool install --from git+ssh://git@github.com/igeniusai/domyn-swarm.git@$RELEASE --python 3.12 domyn-swarm
```

Full instructions, including `uv add`, editable installs and the Lepton
`lep login` step, are in
[Installation](https://domynswarm.domyn.com/latest/getting-started/installation.html).

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

The [Quickstart](https://domynswarm.domyn.com/latest/getting-started/quickstart.html)
walks through the same run with an explanation of each step.

---

## Documentation

The full documentation lives at
**[domynswarm.domyn.com](https://domynswarm.domyn.com/)**.

* [Getting started](https://domynswarm.domyn.com/latest/getting-started/index.html)
  — install, launch a swarm, write your first custom job
* [Guides](https://domynswarm.domyn.com/latest/guides/index.html)
  — Slurm and Lepton, submitting jobs, checkpointing, sharding, data backends,
  swarm state, monitoring
* [Concepts](https://domynswarm.domyn.com/latest/concepts/index.html)
  — architecture, the backend protocols, the `SwarmJob` lifecycle, configuration
  precedence
* Reference, generated from the source on every build —
  [CLI](https://domynswarm.domyn.com/latest/reference/cli.html),
  [Configuration](https://domynswarm.domyn.com/latest/reference/configuration.html),
  [Environment variables](https://domynswarm.domyn.com/latest/reference/environment.html),
  [Python API](https://domynswarm.domyn.com/latest/reference/api/index.html)

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
