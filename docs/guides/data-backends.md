# Choosing a data backend

A **data backend** decides how job input and output are read, written and
iterated. Three are available: `pandas`, `polars` and `ray`.

:::{note}
Unrelated to the *serving* and *compute* backends that decide where the model
runs and where jobs execute — see
[Serving vs compute backends](../concepts/backends.md). "Backend" means two
different things in this project and conflating them causes real confusion.
:::

## The three backends

| Backend | Install | Reach for it when |
| --- | --- | --- |
| `pandas` | always available, the default | the input fits comfortably in memory |
| `polars` | `pip install 'domyn-swarm[polars]'` | the input is large, or you want sharded directory output |
| `ray` | `pip install 'domyn-swarm[ray]'` | you already run a Ray cluster and want distributed execution |

Select one with `--data-backend`. The default is `pandas`, and it is the right
answer more often than not — a batch inference job is usually bound by the
endpoint, not by the dataframe library.

## pandas

Reads the whole input into memory and works with `pd.DataFrame` throughout.

Its practical advantages are worth naming: glob patterns like `data-*.parquet`
work on input, checkpointing uses the well-tested `CheckpointManager` path, and
when something goes wrong the objects in the traceback are ones you already know
how to inspect.

If the input does not fit in memory, that is when to look further.

## polars

Uses `pl.scan_parquet` to build a `LazyFrame` and executes in **streaming** mode,
so the whole input need not be resident. `pl.read_parquet` is used where eager
evaluation is required.

Also the backend to use for sharded directory output: with a directory `--output`
and `--shard-output`, one Parquet file per shard is written using checkpoint state
as the source of truth. See
[Sharding and concurrency](sharding-concurrency.md#output-layout).

## ray

Reads with `ray.data.read_parquet` and distributes execution across a Ray
cluster.

Two hard requirements:

- **`--ray-address`** pointing at the cluster
- **native execution.** Ray requires it, and passing `--no-native-backend`
  raises:

  ```text
  Ray backend requires native execution (native_backend=True).
  ```

Ray does not use `--num-shards` sharding — distribution is Ray's own concern.

Worth being honest about the trade: Ray adds a second scheduler underneath a
system that already has one. If your bottleneck is the model endpoint rather than
data processing, Ray adds operational complexity without adding throughput. It
earns its place when the per-row work around the model call is itself heavy, or
when a Ray cluster is already part of your environment.

## How one gets selected

`_resolve_backend_name` resolves in order:

1. the `--data-backend` flag
2. a `data_backend` attribute on the job class
3. `pandas`

So a job that only ever makes sense on one backend can declare it, and the flag
still overrides.

`get_backend(name)` then constructs it, importing polars or Ray lazily so the
default install never pays for them.

## When an extra is missing

The error names the fix:

```text
Polars backend requires `polars` to be installed.
Ray backend requires `ray[data]` to be installed.
```

Both are `BackendError`, raised from the underlying `ImportError`. An unknown
name gives `Unknown data backend: <name>`. If you see one of these, install the
[matching extra](../getting-started/installation.md#optional-extras).

## Runners

`--runner` selects `pandas` or `arrow` for the non-Ray backends. The Arrow runner
keeps data as `pyarrow` tables rather than converting to pandas, which avoids
conversion cost and memory spikes on wide or large batches. `pandas` remains the
default.

`--native-batch-size` sets the batch size in native backend mode, used by Ray and
polars. `--backend-read-kwargs` and `--backend-write-kwargs` take JSON objects
forwarded to the backend's own read and write calls, which is how you reach
options this CLI does not wrap — compression, row-group size, storage options.

## What a backend must provide

To add a fourth, implement the `DataBackend` protocol:

`read` / `write`
: load and persist a dataset, honouring `limit` and `nshards`

`to_pandas` / `from_pandas`, `to_arrow` / `from_arrow`
: convert to and from the two interchange formats

`schema`, `slice`, `iter_batches`
: describe, subset and chunk the data

`iter_job_batches`
: yield `JobBatch` objects — the important one

`JobBatch` is the normalisation point that keeps the rest of the system
backend-agnostic. Each carries `ids` (stable identifiers for checkpointing and
resume), `items` (extracted from the input column, passed to `transform_items`),
and `batch` (the backend-native object, for downstream joins and debugging).

Because every backend produces the same `JobBatch`, nothing in the execution or
checkpointing path needs to know which library is underneath.
