# Getting started

Three pages, in order. Start to finish is about fifteen minutes on a cluster you
already have access to.

- [Installation](installation.md) — the package, the optional extras, and what
  each one is for
- [Quickstart](quickstart.md) — launch a swarm, run a batch job against it, tear
  it down
- [Your first custom job](first-custom-job.md) — write a `SwarmJob` subclass and
  run it from both the CLI and a script

Once those work, [Guides](../guides/index.md) covers doing it properly:
checkpointing long runs, choosing a data backend, and the platform-specific
details for Slurm and Lepton.

```{toctree}
:hidden:

installation
quickstart
first-custom-job
```
