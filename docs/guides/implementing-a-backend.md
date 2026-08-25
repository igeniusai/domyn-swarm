# Implementing a backend

Adding a platform — another cloud, another scheduler — means writing two classes
and a config model. Nothing in the job layer, the CLI or the state database needs
to change, which is the point of splitting serving from compute in the first
place: see [Serving vs compute backends](../concepts/backends.md) for why the
seam is drawn where it is.

This is a guide for contributors. It describes internal interfaces, which are not
covered by the public API promise in [Reference](../reference/index.md) and can
change between releases.

## The shape of it

Three pieces, and a fourth line to register them:

1. a **serving backend** that creates an endpoint and reports when it is ready
2. a **compute backend** that runs a job against that endpoint
3. a **config model** carrying your platform's settings, which builds the pair
4. an entry in the backend union so `type: yours` in YAML selects it

Both backends are `typing.Protocol` definitions, not base classes. You do not
inherit from them — implementing the methods is enough, and
`isinstance` still works because they are `runtime_checkable`.

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

Put everything reattachment needs in `meta`. The process that ran `up` is gone by
the time someone runs `job submit --name my-swarm`, and `meta` is how your backend
recognises what it created — this is what makes
[swarm state](swarm-state.md) work.

`status` returns a `ServingStatus(phase, url, detail)`, where `phase` is a
`ServingPhase`: `UNKNOWN`, `PENDING`, `INITIALIZING`, `RUNNING`, `FAILED` or
`STOPPED`. Map your platform's vocabulary onto those six — `domyn-swarm status`
and the JSON contract are written against the enum, not your strings, and
anything platform-specific belongs in `detail`.

`create_or_update` is named for a reason: it must be safe to call against a name
that already exists, reconciling rather than failing or duplicating.

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

The mixin's defaults are deliberately conservative: the current interpreter, no
image, no resources, no extra environment. Override the `default_*` hooks when
your platform can infer better, so that a job submission does not have to name
an interpreter or an image the platform already knows.

`JobStatus` is the five-state vocabulary — `PENDING`, `RUNNING`, `SUCCEEDED`,
`FAILED`, `CANCELLED`. Use `coerce_job_status()` to normalise a raw payload rather
than mapping strings yourself; it falls back to `PENDING` on anything it does not
recognise.

### probe versus wait

`wait` blocks. `probe` must not: it answers "what is this job doing right now",
and it is what `job status --refresh` calls. Returning a `JobProbe` with `error`
set is how you say *I could not tell* — which is not the same as *it failed*, and
the distinction matters, because [`db prune`](swarm-state.md) removes records
whose probe raises.

The mixin's `probe` just echoes the handle's last known status with
`source="local"`. That is honest but useless for a platform you can actually
query, so implement it if you can.

### Sharding

`nshards` and `shard_id` are passed through to your `submit`. You do not have to
schedule shards yourself — each shard arrives as its own `submit` call. Just make
sure both values reach the job process, since the job layer uses them to decide
which rows it owns. See [Sharding and concurrency](sharding-concurrency.md).

## 3. The config model

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

Two conventions worth following rather than discovering:

**Describe every field.** `Field(description=...)` is what the
[configuration reference](../reference/configuration.md) renders, and a test
fails on any field without one. It walks the config graph from the root, so a
model reachable from your config is covered automatically.

**Import platform SDKs inside `build`.** Every backend does this. Loading a YAML
config must not require the SDKs of platforms you are not using — the Lepton SDK
is an optional extra precisely so Slurm users need not install it.

`cfg_ctx` is the whole `DomynLLMSwarmConfig`, which is where the
platform-independent settings live: `replicas`, `gpus_per_replica`, `model`,
`env`. Your `serving_spec` is the dict handed to `create_or_update`, so merge in
whatever your backend needs from both.

## 4. Register it

Add the model to the discriminated union in
[`config/backend.py`](https://github.com/igeniusai/domyn-swarm/blob/main/src/domyn_swarm/config/backend.py):

```python
BackendConfig = Annotated[
    LeptonConfig | SlurmConfig | MyBackendConfig,
    Field(discriminator="type"),
]
```

That is the whole registration. `DomynLLMSwarmConfig.backend` is typed as
`BackendConfig`, so `type: mybackend` in YAML now selects your model, `PlanBuilder`
calls your `build`, and the CLI works unchanged.

```yaml
model: "some-org/some-model"
replicas: 2
backend:
  type: mybackend
  region: eu-west-1
```

## What you will hit

**`DeploymentPlan.platform` is a closed literal.** It is typed
`Literal["lepton", "slurm"]`, so a third platform needs that annotation widened.
It is also worth checking where `platform` is read before assuming a new value is
handled everywhere.

**Not every feature is backend-agnostic.** Some deliberately are not, and it is
better to know which up front than to discover a half-working feature:

| Feature | Where it lives |
| --- | --- |
| Watchdog and collector | Slurm only, built around the array-job layout |
| [Prometheus monitoring](metrics.md) | Slurm only, built out of load-balancer sidecars |
| Swarm state, jobs, checkpointing, data backends | platform-independent |

A new backend gets the whole job layer for free, and gets no health reporting
until someone writes it. That is not a bug in your backend — `status` reporting
`RUNNING` from the platform with no per-replica rows is the expected shape for a
backend that has no watchdog.

## Testing it

Both protocols are `runtime_checkable`, so the cheapest useful test asserts you
actually satisfy them:

```python
from domyn_swarm.platform.protocols import ComputeBackend, ServingBackend

def test_backends_satisfy_the_protocols():
    assert isinstance(MyServingBackend(cfg=cfg), ServingBackend)
    assert isinstance(MyComputeBackend(cfg=cfg), ComputeBackend)
```

Be aware of what that does *not* check: `runtime_checkable` protocols verify
method *names*, not signatures. A `submit` with the wrong keyword arguments passes
`isinstance` and fails when called, so test a real `submit` and `probe` against a
faked platform client as well.

The existing backends under `src/domyn_swarm/backends/` are the reference
implementations, and `tests/backends/` shows how they are tested without a
cluster.
