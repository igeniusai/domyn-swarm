# Sharding and concurrency

## Two independent dials

Two independent knobs, easily confused:

`--max-concurrency`
: how many requests are in flight **at once**, within a shard

`--num-shards`
: how many **shards** the input is split into

They multiply. Four shards at concurrency 8 is up to 32 simultaneous requests
against your endpoint. Tuning one without noticing the other is the usual reason
a swarm gets overwhelmed.

## Bounded concurrency

`--max-concurrency` is enforced with an `asyncio.Semaphore` and a matching worker
count, so it is a hard ceiling on in-flight requests, not a target.

Raise it until the endpoint is saturated and no further. Past that point extra
concurrency adds queueing latency and makes timeouts more likely without adding
throughput — the replicas are already busy.

## Retries and timeouts

`--retries` sets the maximum attempts per item, applied with tenacity's
exponential backoff: multiplier 1, minimum 4 seconds, capped at 10.

`--timeout` is the per-request timeout in seconds. It interacts with concurrency:
if concurrency is high enough that requests queue at the endpoint, a timeout
tuned for an idle endpoint will start firing under load. Symptoms of that are
covered in [Monitoring and troubleshooting](monitoring.md).

## Sharding the input

`--num-shards` splits the input into that many shards, run concurrently, each
with its own checkpoint directory. That per-shard isolation is what makes
concurrent flushing safe.

:::{important}
Sharded execution **requires checkpointing**. Combining `--num-shards > 1` with
`--no-checkpointing` raises:

```text
Sharded execution requires checkpointing to be enabled.
```

Shard results are assembled from checkpoint state, so there is nothing to
assemble without it.
:::

## Choosing a shard mode

`--shard-mode` takes exactly `id` or `index`; anything else raises a `ValueError`.

`id` (default)
: assigns each row by a stable hash of its id — `hash_array(ids) % nshards`. A
  given row lands in the same shard regardless of where it sits in the input, or
  how long the input is.

`index`
: splits row order into contiguous blocks with `np.array_split`. This is the
  legacy behaviour.

The consequence matters for resume. With `index`, inserting a row near the start
shifts everything after it into different shards, and previously completed work
becomes invisible to the shard now responsible for it. With `id`, row identity
survives reordering and length changes.

Prefer `id` unless you specifically need contiguous blocks — and pair it with a
real [`--id-column`](submitting-jobs.md#row-identity), since a generated index
is not stable across scans.

## Resume across a changed shard count

Even with `id`, changing `--num-shards` changes `nshards` and therefore the
modulo, so rows move between shards. A shard will not find another shard's
checkpoint.

Two ways through:

- keep `--num-shards` fixed across resumed runs, which is the simple answer
- pass `--global-resume`, which filters the input against the union of done ids
  across all shards rather than each shard's own

The same applies to `--limit`, since it changes which rows are present.
See [Checkpointing and resuming](checkpointing.md).

## Output layout

Checkpoint stores are per shard, derived by inserting the shard index:
`run.parquet` becomes `run_shard0.parquet`, `run_shard1.parquet` and so on.

For the final output, a single-file `--output` receives all shards merged. With a
directory output plus `--shard-output` on the Polars runner, one Parquet file per
shard is written instead, named zero-padded by shard count — `data-00.parquet`,
`data-01.parquet` — so lexical order matches shard order. Checkpoint outputs are
the source of truth for that write.

Directory outputs are the better choice for large results: no merge step, and
downstream readers can consume the shards in parallel.

## Ray

Ray takes a different execution path and does not use `--num-shards` sharding —
distribution is Ray's own concern. It requires `--native-backend` and a
`--ray-address`. See [Choosing a data backend](data-backends.md).

## A starting point

For a first run on an unfamiliar model and cluster:

```bash
--max-concurrency 8 --num-shards 1 --retries 3 --timeout 600 --limit 100
```

Confirm correctness on 100 rows, then raise `--max-concurrency` until throughput
stops improving, then add shards with `--num-shards` if the driver process
itself becomes the bottleneck rather than the endpoint.
