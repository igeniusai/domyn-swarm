# domyn-swarm

A CLI and Python library for launching **LLM serving endpoints** and running
**high-throughput batch jobs** against them, with first-class support for
**Slurm** and **NVIDIA DGX Cloud Lepton**.

```bash
domyn-swarm up -c config.yaml
domyn-swarm job submit --name my-swarm --input prompts.parquet --output answers.parquet
domyn-swarm down my-swarm
```

## Why domyn-swarm

Serving a model at scale and running work against it are usually two unrelated
chores: one person writes the launch scripts, another writes the batch loop, and
the two agree by convention about where the endpoint lives. domyn-swarm makes
that one workflow. A YAML file describes the swarm; `up` stands up replicas
behind a load balancer and waits until they actually answer; `job submit` runs a
typed job against them with batching, retries and checkpointing; `down` removes
everything it created.

The same commands work on an HPC cluster and in the cloud. Only the config's
`backend` section changes.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket` Get started
:link: getting-started/index
:link-type: doc

Install, launch your first swarm, and write your first custom job.
:::

:::{grid-item-card} {octicon}`book` Guides
:link: guides/index
:link-type: doc

Checkpointing, sharding, data backends, and the platform-specific details.
:::

:::{grid-item-card} {octicon}`light-bulb` Concepts
:link: concepts/index
:link-type: doc

How the pieces fit together, and why they are split the way they are.
:::

:::{grid-item-card} {octicon}`terminal` Reference
:link: reference/index
:link-type: doc

CLI, configuration and Python API, generated from the source.
:::

::::

## What it does

- **One CLI across platforms** — `up`, `job submit`, `status`, `down` behave the
  same on Slurm and Lepton
- **Endpoints that are ready when they say they are** — replicas are health-probed
  before `up` returns, so a job never starts against a model that is still loading
- **Jobs that survive failure** — DataFrame in, DataFrame out, with bounded
  concurrency, retries with backoff, and Parquet checkpointing that can resume a
  half-finished run
- **A script escape hatch** — when a job class is the wrong shape, submit any
  Python file to the compute backend instead
- **Swarms you can find again** — state is kept in a local SQLite database, so a
  swarm is addressable by name from any later command
- **Backends behind protocols** — serving and compute are separate interfaces, so
  supporting a new platform adds code rather than changing it

## Supported backends

Serving and compute — where the model runs, and where jobs execute:

| Backend | Notes |
| --- | --- |
| **Slurm** | Singularity containers and a job array for replicas, behind an Nginx load balancer |
| **NVIDIA DGX Cloud Lepton** | Endpoint plus batch job through the Lepton SDK (`domyn-swarm[lepton]`) |

Data — how job input and output are read, written and iterated. See
[Choosing a data backend](guides/data-backends.md):

| Backend | Install |
| --- | --- |
| **pandas** | always available; the default |
| **polars** | `domyn-swarm[polars]` |
| **ray** | `domyn-swarm[ray]` |

## Where to go next

New here? [Installation](getting-started/installation.md) then
[Quickstart](getting-started/quickstart.md) is about fifteen minutes end to end.

Already running jobs? [Checkpointing and resuming](guides/checkpointing.md) and
[Sharding and concurrency](guides/sharding-concurrency.md) are the two guides
that most change how a long run behaves.

Debugging something? [Monitoring and troubleshooting](guides/monitoring.md)
starts from the symptom.

```{toctree}
:hidden:

getting-started/index
guides/index
concepts/index
reference/index
```
