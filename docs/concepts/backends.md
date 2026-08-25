# Serving vs compute backends

:::{note}
"Backend" means two unrelated things in domyn-swarm. This page is about
**serving** and **compute** backends, which decide *where the model runs and
where jobs execute*. A **data** backend decides how job input and output are read
and written — that is [Choosing a data backend](../guides/data-backends.md).
Readers conflate these constantly.
:::

## Why two protocols instead of one backend type

A single `Backend` abstraction would have to answer both "how do I serve a model
here" and "how do I run a process here", and those questions have different
shapes and different failure modes. Splitting them buys three things:

- **Mixing.** Serving and compute are chosen independently, so a model served in
  one place can be driven by work executing somewhere else.
- **Independent evolution.** Adding a readiness strategy touches only the serving
  side; adding a way to launch processes touches only compute.
- **Honest health reporting.** "The endpoint is ready" and "the job succeeded"
  are separate states with separate owners, which is why
  `domyn-swarm status` can report a healthy endpoint and a failed job at once.

Both are `typing.Protocol` definitions, so a backend conforms structurally — no
inheritance required, and no base class to fight when a platform does something
unusual.

## ServingBackend

Owns the model endpoint and its lifetime.

| Method | Responsibility |
| --- | --- |
| `create_or_update(name, spec, extras)` | Create the endpoint or reconcile an existing one; returns a `ServingHandle` |
| `wait_ready(handle, timeout_s, extras)` | Block until the endpoint can serve |
| `ensure_ready(handle)` | Assert readiness for an already-created endpoint |
| `status(handle)` | Report a `ServingStatus`, carrying a `ServingPhase` |
| `delete(handle)` | Remove the endpoint |

`create_or_update` rather than `create` is deliberate: re-running `up` against an
existing swarm reconciles instead of failing or duplicating.

## ComputeBackend

Owns the processes that call the endpoint.

| Method | Responsibility |
| --- | --- |
| `submit(...)` | Start a job; returns a `JobHandle` |
| `wait(handle, stream_logs=True)` | Block until completion; returns a `JobStatus` |
| `cancel(handle)` | Stop a running job |
| `probe(handle)` | Report a `JobProbe`, which is how `job status --refresh` works |
| `default_python(cfg)` | Interpreter to run the job with |
| `default_image(cfg)` | Container image, where the platform uses one |
| `default_resources(cfg)` | Platform resource request |
| `default_env(cfg)` | Environment the job needs |

The four `default_*` methods keep platform knowledge out of the submission path:
the caller does not need to know that Slurm wants a `venv_path` while Lepton
wants a Docker image. `DefaultComputeMixin` supplies the common implementations,
including a `probe` that most backends can use unchanged.

## How the two platforms satisfy them

**Slurm.** Serving is an array job of vLLM replicas plus an Nginx load-balancer
job; readiness is an HTTP probe against `/v1/health` through the load balancer.
Compute is `srun` into the allocation, with `require_allocated_node` guarding
against work landing on the load-balancer node.

**Lepton.** Serving is a Lepton endpoint, which fronts its own replicas, so there
is no load balancer to manage; readiness is deployment-state polling rather than
an HTTP probe. Compute is a Lepton batch job.

Notice that the *readiness strategies differ entirely* while the protocol does
not. That is the separation earning its keep.

## Adding a platform

The README describes new targets as "easy to add", which is worth qualifying.
What the protocols genuinely give you is a closed list of methods and no base
class to satisfy — the interface is small and the compiler-checkable part is
straightforward.

What they do not remove is the real work: readiness semantics, log retrieval,
identifier mapping into `ServingHandle` and `JobHandle`, cancellation that
actually stops things, and the config model plus its `type` discriminator. Look at
`backends/serving/` and `backends/compute/` for either existing platform to
gauge it honestly — the protocol is a day, the semantics are not.

For the step-by-step version — the method sets, the handle contract, the
registration line, and which features are not backend-agnostic — see
[Implementing a backend](../guides/implementing-a-backend.md).
