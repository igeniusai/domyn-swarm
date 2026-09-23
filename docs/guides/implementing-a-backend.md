# Implementing a backend

Adding a cloud or scheduler requires two backend classes and one configuration
model. The job layer, CLI, and state database do not change. See
[Serving vs compute backends](../concepts/backends.md) for this boundary.

This is a guide for contributors. It describes internal interfaces, which are not
covered by the public API promise in [Reference](../reference/index.md) and can
change between releases.

## The shape of it

Implement and register these components:

1. A serving backend that creates an endpoint and reports readiness
2. A compute backend that runs a job against the endpoint
3. A configuration model that builds the backend pair
4. A backend union entry that selects the model from its YAML `type`

Both backends are `typing.Protocol` definitions, not base classes. Implement
their methods without inheritance. `isinstance` works because the protocols are
`runtime_checkable`.

## 1. The serving backend

Create the endpoint, wait for it, report on it, delete it. From
`domyn_swarm.platform.protocols`:

```python
class ServingBackend(Protocol):
    def create_or_update(self, name: str, spec: dict, extras: dict) -> ServingHandle: ...
    def wait_ready(self, handle: ServingHandle, timeout_s: int, extras: dict) -> ServingHandle: ...
    def ensure_ready(self, handle: ServingHandle): ...
    def status(self, handle: ServingHandle) -> ServingStatus: ...
    def delete(self, handle: ServingHandle) -> None: ...
```

A `ServingHandle` is what gets persisted, so it must be enough to find the
endpoint again in a *later process*:

```python
@dataclass
class ServingHandle:
    id: str                  # your platform's identifier
    url: str                 # base URL to call; empty until ready
    meta: dict[str, Any]     # ports, job ids, workspace, whatever you need back
```

Put all reattachment data in `meta`. The original `up` process has ended when a
later command uses `job submit --name my-swarm`. The backend uses `meta` to find
its resources. See [swarm state](swarm-state.md).

`status` returns a `ServingStatus(phase, url, detail)`, where `phase` is a
`ServingPhase`: `UNKNOWN`, `PENDING`, `INITIALIZING`, `RUNNING`, `FAILED` or
`STOPPED`. Map platform states to these six values. `domyn-swarm status` and the
JSON contract use this enum. Put platform-specific information in `detail`.

`create_or_update` must reconcile an existing name without failure or
duplication.

## 2. The compute backend

```python
class ComputeBackend(Protocol):
    def submit(self, *, name: str, image: str | None, command: Sequence[str],
               env: Mapping[str, str] | None = None, resources: dict | None = None,
               detach: bool = False, nshards: int | None = None,
               shard_id: int | None = None, extras: dict | None = None) -> JobHandle: ...
    def wait(self, handle: JobHandle, *, stream_logs: bool = True) -> JobStatus: ...
    def cancel(self, handle: JobHandle) -> None: ...
    def probe(self, handle: JobHandle) -> JobProbe: ...
    def default_python(self, cfg) -> str: ...
    def default_image(self, cfg) -> str | None: ...
    def default_resources(self, cfg) -> dict | None: ...
    def default_env(self, cfg) -> dict[str, str]: ...
```

Inherit `DefaultComputeMixin` and you only owe the first three:

```python
from domyn_swarm.platform.protocols import DefaultComputeMixin

class MyComputeBackend(DefaultComputeMixin):
    def submit(self, *, name, command, image=None, env=None, **kw): ...
    def wait(self, handle, *, stream_logs=True): ...
    def cancel(self, handle): ...
```

The mixin uses the current interpreter without an image, resources, or extra
environment. Override a `default_*` hook when the platform can infer a better
value.

`JobStatus` contains `PENDING`, `RUNNING`, `SUCCEEDED`, `FAILED`, and `CANCELLED`.
Use `coerce_job_status()` to normalize a raw payload. Unknown values become
`PENDING`.

### probe versus wait

`wait` blocks, but `probe` must return immediately. `job status --refresh` calls
`probe`. Set `JobProbe.error` when the backend cannot determine the state. This
state is different from a failed job. [`db prune`](swarm-state.md) removes
records when a probe raises an exception.

The mixin returns the last recorded status with `source="local"`. Implement
`probe` when the platform provides live job state.

### Sharding

`nshards` and `shard_id` pass through to `submit`. Each shard has a separate
`submit` call. Make sure that both values reach the job process because the job
layer uses them to select rows. See
[Sharding and concurrency](sharding-concurrency.md).

## 3. The configuration model

A Pydantic model with a literal `type`, which is the discriminator, and a `build`
that returns the assembled pair:

```python
from typing import Literal
from pydantic import BaseModel, Field
from domyn_swarm.config.plan import DeploymentPlan


class MyBackendConfig(BaseModel):
    type: Literal["mybackend"] = Field(
        default="mybackend",
        description="Backend discriminator; always `mybackend` for this model.",
    )
    region: str = Field(description="Where to create the endpoint.")

    def build(self, cfg_ctx) -> DeploymentPlan:
        # Import backends inside build, not at module scope: a platform SDK must
        # not be imported by anyone who merely loads a config.
        from mypackage.backends import MyComputeBackend, MyServingBackend

        return DeploymentPlan(
            name_hint="mybackend",
            serving=MyServingBackend(cfg=self),
            compute=MyComputeBackend(cfg=self),
            serving_spec=self.model_dump(exclude_none=True),
            job_resources={},
            extras={},
        )
```

Follow these conventions:

1. Add `Field(description=...)` to every field. The
   [configuration reference](../reference/configuration.md) renders this text.
   A test fails when a reachable model contains a field without a description.

2. Import platform SDKs inside `build`. Loading a YAML configuration must not
   import an unused platform SDK. For example, Slurm users do not install the
   optional Lepton SDK.

`cfg_ctx` contains the full `DomynLLMSwarmConfig`, including `replicas`,
`gpus_per_replica`, `model`, and `env`. `create_or_update` receives
`serving_spec`. Add all required values from both configuration objects.

## 4. Register it

Add the model to the discriminated union in
[`config/backend.py`](https://github.com/igeniusai/domyn-swarm/blob/main/src/domyn_swarm/config/backend.py):

```python
BackendConfig = Annotated[
    LeptonConfig | SlurmConfig | MyBackendConfig,
    Field(discriminator="type"),
]
```

`DomynLLMSwarmConfig.backend` uses the `BackendConfig` union. A YAML value of
`type: mybackend` selects the new model. `PlanBuilder` then calls `build`. The
CLI does not need another registration step.

```yaml
model: "some-org/some-model"
replicas: 2
backend:
  type: mybackend
  region: eu-west-1
```

## What you will hit

`DeploymentPlan.platform` is a closed literal. It uses
`Literal["lepton", "slurm"]`. Add the new platform to this annotation and inspect
each consumer of `platform`.

Some features are platform-specific:

| Feature | Where it lives |
| --- | --- |
| Watchdog and collector | Slurm only, built around the array-job layout |
| [Prometheus monitoring](metrics.md) | Slurm only, built out of load-balancer sidecars |
| Swarm state, jobs, checkpointing, data backends | platform-independent |

A new backend can use the existing job layer. It has no replica health data
until it implements health reporting. Without a watchdog, `status` can report
platform state as `RUNNING` without per-replica rows.

## Testing it

Both protocols are `runtime_checkable`, so the cheapest useful test asserts you
actually satisfy them:

```python
from domyn_swarm.platform.protocols import ComputeBackend, ServingBackend

def test_backends_satisfy_the_protocols():
    assert isinstance(MyServingBackend(cfg=cfg), ServingBackend)
    assert isinstance(MyComputeBackend(cfg=cfg), ComputeBackend)
```

`runtime_checkable` protocols inspect method names, not signatures. A `submit`
method with incorrect keyword arguments can pass `isinstance`. Test `submit`
and `probe` with a fake platform client.

The existing backends under `src/domyn_swarm/backends/` are the reference
implementations, and `tests/backends/` shows how they are tested without a
cluster.
