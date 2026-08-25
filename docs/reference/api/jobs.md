# Jobs

## SwarmJob

Subclass this and implement `transform_items` to define a job. Batching, bounded
concurrency, retries and checkpointing are provided by the framework.

:::{note}
Import `SwarmJob` from the `domyn_swarm` package root. The
`domyn_swarm.jobs.base` module is a deprecated alias for
`domyn_swarm.jobs.api.base`, which is what is documented here.
:::

```{eval-rst}
.. autoclass:: domyn_swarm.jobs.api.base.SwarmJob
   :members:
   :show-inheritance:
```

## run_job_unified

```{eval-rst}
.. autofunction:: domyn_swarm.jobs.execution.dispatch.run_job_unified
```
