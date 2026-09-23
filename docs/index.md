# domyn-swarm

A CLI and Python library for launching LLM serving endpoints and running
high-throughput batch jobs against them. It supports Slurm and NVIDIA DGX Cloud
Lepton.

```bash
domyn-swarm up -c config.yaml
domyn-swarm job submit --name my-swarm --input prompts.parquet --output answers.parquet
domyn-swarm down my-swarm
```

## Why domyn-swarm

Model serving and batch processing often use separate tools. domyn-swarm joins
them in one workflow. A YAML file describes the swarm. `up` starts replicas
behind a load balancer and waits for them to respond. `job submit` runs a typed
job with batching, retries, and checkpointing. `down` removes the deployment.

The same commands work on an HPC cluster and in the cloud. Only the configuration's
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

- One CLI supports Slurm and Lepton. `up`, `job submit`, `status`, and `down`
  use the same interface on both platforms.
- Health probes run before `up` returns. Jobs do not start while a model is still
  loading.
- Jobs support bounded concurrency, retries with backoff, and Parquet
  checkpoints. An interrupted job can resume from completed work.
- `job submit-script` submits a Python file when a job class does not fit the
  task.
- A local SQLite database stores swarm state. Later commands can address the
  swarm by name.
- Separate serving and compute protocols isolate platform-specific code.

## Supported backends

Serving and compute backends control where models and jobs run:

| Backend | Notes |
| --- | --- |
| Slurm | Singularity containers and a job array for replicas, behind an Nginx load balancer |
| NVIDIA DGX Cloud Lepton | Endpoint plus batch job through the Lepton SDK (`domyn-swarm[lepton]`) |

Data backends control how jobs read, write, and iterate over data. See
[Choosing a data backend](guides/data-backends.md):

| Backend | Install |
| --- | --- |
| pandas | always available and used by default |
| polars | `domyn-swarm[polars]` |
| ray | `domyn-swarm[ray]` |

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
