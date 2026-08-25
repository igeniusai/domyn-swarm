# Guides

Task-oriented, and mostly independent of one another — read the one you need.

## Platforms

- [Running on Slurm](slurm.md) — Singularity images, bind mounts, modules, node selection
- [Running on Lepton](lepton.md) — endpoint and job config, secrets, and how it differs from Slurm

## Running work

- [Submitting jobs](submitting-jobs.md) — input formats, job classes, the flags that matter
- [Checkpointing and resuming](checkpointing.md) — surviving a failed run
- [Sharding and concurrency](sharding-concurrency.md) — the two dials, and how they affect resume
- [Choosing a data backend](data-backends.md) — pandas, polars or Ray

## Operating

- [Managing swarm state](swarm-state.md) — the state database, `db` commands, inspecting swarms
- [Monitoring and troubleshooting](monitoring.md) — reading `status`, and what to do when a replica is unhealthy
- [Metrics and dashboards](metrics.md) — Prometheus, GPU exporters and `domyn-swarm monitor`
- [Swarm pools](swarm-pools.md) — incomplete; read before trying to use it

## Extending

- [Implementing a backend](implementing-a-backend.md) — adding a platform: the two
  protocols, the config model, and which features are not backend-agnostic

For *why* things are built this way rather than how to use them, see
[Concepts](../concepts/index.md). For exhaustive flag and field lists, see
[Reference](../reference/index.md).

```{toctree}
:hidden:

slurm
lepton
submitting-jobs
checkpointing
sharding-concurrency
data-backends
swarm-pools
swarm-state
monitoring
metrics
implementing-a-backend
```
