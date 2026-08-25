# Architecture

Serving a model and running work against it are two different problems, and
domyn-swarm keeps them apart. That separation is the single idea most worth
understanding before changing anything.

## DomynLLMSwarm owns the lifecycle

{py:class}`~domyn_swarm.core.swarm.DomynLLMSwarm` is a context manager. Entering
it brings an endpoint up and waits for readiness; leaving it tears the endpoint
down, unless `delete_on_exit=False` keeps the allocation alive for later.

```python
with DomynLLMSwarm(cfg=cfg) as swarm:
    swarm.submit_job(job, input_path="in.parquet", output_path="out.parquet")
```

It is a thin coordinator. The platform-specific work lives below it.

## Deployment pairs a serving backend with a compute backend

`Deployment` composes exactly two collaborators and nothing else:

- a **`ServingBackend`**, which owns the model endpoint
- a **`ComputeBackend`**, which owns the processes that call that endpoint

Its flow is `up(name, ctx)` → `wait_ready(timeout_s)` → `run(...)` → `down(handle)`,
and it is itself a context manager, so a raised exception still deletes the
endpoint rather than leaking an allocation.

The handle passed between those calls is a `ServingHandle`, carrying the endpoint
URL and whatever platform metadata the backend needs to find its resources again.

## What each side is responsible for

The serving side provides:

`create_or_update`
: create the endpoint, or reconcile an existing one, returning a `ServingHandle`

`wait_ready`
: block until the endpoint can serve requests

`status`
: report a `ServingStatus`, which is what `domyn-swarm status` surfaces

`delete`
: remove the endpoint

The compute side provides:

`submit`
: start a job, returning a `JobHandle`

`wait`
: block until the job finishes, returning a `JobStatus`

`cancel`
: stop a running job

`probe`
: report a `JobProbe` for a handle, which is how status refresh works

Compute backends also supply defaults — `default_python`, `default_image`,
`default_resources`, `default_env` — so that a job submission does not have to
name a Python interpreter or an image the platform can infer. `DefaultComputeMixin`
provides the common answers.

Both are `Protocol` definitions rather than base classes, so a backend satisfies
them structurally. See
[Serving vs compute backends](backends.md).

## Jobs run beside the endpoint, not inside it

A job never runs in the serving container. The compute backend starts a separate
process — `srun` on Slurm, a batch job on Lepton — with `ENDPOINT` and `MODEL` in
its environment, and that process talks to the endpoint over HTTP like any other
client.

This is why `job submit-script` can run arbitrary Python: from the endpoint's
point of view there is no difference between a `SwarmJob` and a script you wrote,
and adding a new job type needs no change to the serving side.

See [The SwarmJob lifecycle](swarmjob-lifecycle.md).

## State makes swarms outlive processes

Every swarm is recorded in a local SQLite database: deployment metadata, resource
handles, the configuration it was created from, platform identifiers such as job
IDs and node assignments, and the endpoint URL.

That record is why `--name` works. The process that ran `up` is long gone by the
time you run `job submit --name my-swarm`; the swarm is rehydrated from state
rather than re-derived. It is also what makes `DomynLLMSwarm.from_state(name)`
possible, and what `swarm list` reads.

See [Managing swarm state](../guides/swarm-state.md).

## The load balancer reconciles rather than being configured

Nginx needs to know where the replicas are, but nothing knows that at submission
time — Slurm decides placement, and replicas appear one by one. So the
load-balancer config is not written once; it converges.

Each replica writes a `replica-<id>.head` file, holding its `host:port`, into the
swarm's shared serving directory. A **supervisor** process watches that directory
and regenerates `00-upstreams.conf` from whatever is currently there, plus
Prometheus's target files when monitoring is on. Adding or losing a replica is
therefore a file appearing or vanishing, not an event anyone has to deliver.

The supervisor only *writes*. It never reloads Nginx — and that split is forced
rather than chosen. Nginx runs in its own Singularity instance, and
`singularity instance start` puts it in a private PID namespace, so a process in
another container cannot signal the Nginx master. Instead the load-balancer script
watches the generated file host-side and, when it changes, runs `nginx -t` and
then `nginx -s reload` through `singularity exec instance://`. A config that
fails validation is not loaded, so a partial write cannot take the endpoint down.

The same reconcile loop is what makes metrics work without configuration: the
targets Prometheus reads are generated from the same head files as the upstreams,
which is why a new replica is scraped without anyone editing a scrape config. See
[Metrics and dashboards](../guides/metrics.md).

## Health is reported, not inferred

Replicas do not simply run and hope. Each is supervised by a watchdog that probes
it and reports to a single collector, which owns the health database that
`domyn-swarm status` reads. That indirection exists for a specific reason,
explained in [Watchdog and collector](watchdog-collector.md).

## Where to go next

- [Serving vs compute backends](backends.md) — the two protocols in detail
- [The SwarmJob lifecycle](swarmjob-lifecycle.md) — from CLI to output file
- [Watchdog and collector](watchdog-collector.md) — why health has its own process
- [Configuration precedence](configuration.md) — where a value actually comes from
