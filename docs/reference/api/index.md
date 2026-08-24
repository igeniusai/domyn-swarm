# Python API

The public surface is the four names exported from `domyn_swarm`:

```python
from domyn_swarm import DomynLLMSwarm, DomynLLMSwarmConfig, SwarmJob, run_job_unified
```

Everything else under `domyn_swarm.*` is internal and may change without notice.

:::{warning}
This API is still evolving; expect breaking changes before the stable release.
Legacy `transform(df)`-based jobs are no longer supported — implement
`transform_items(items)`, or rely on the `transform_streaming` provided by
`SwarmJob`.
:::

```{toctree}
:maxdepth: 2

swarm
jobs
```
