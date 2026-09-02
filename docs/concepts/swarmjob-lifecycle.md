# The SwarmJob lifecycle

What happens between typing `domyn-swarm job submit` and a Parquet file appearing.
Understanding this is mostly understanding how little your own code has to do.

## 1. The CLI resolves the class

`job submit` takes `<module>:<ClassName>` and imports it, defaulting to
`domyn_swarm.jobs:ChatCompletionJob`. `--job-kwargs` is parsed as JSON and passed
to the constructor as configuration overrides. Provider parameters that the
CLI doesn't need to understand — `temperature`, `top_p`, and the like — go
under `request_params`, so they reach the client without the CLI knowing what
they mean:

```json
{"max_concurrency": 8, "request_params": {"temperature": 0.2}}
```

The swarm is located either by `--name`, rehydrated from the state record, or
created fresh from `--config`.

## 2. A driver process starts beside the endpoint

The compute backend submits a process — `srun` on Slurm, a batch job on Lepton —
with `ENDPOINT` and `MODEL` set in its environment. That process runs
`python -m domyn_swarm.jobs.cli.run`.

The job does **not** execute inside the serving container. It is an HTTP client
like any other.

## 3. The API version is resolved

`resolve_job_api` decides which execution path applies, in this order:

1. a class-level `api_version >= 2` means the current API
2. otherwise, overriding `transform_items` or `transform_streaming` means the
   current API
3. otherwise, overriding `transform` or `run` means the legacy API
4. otherwise, legacy

Legacy `transform(df)` jobs are **no longer supported** — `_ensure_new_api`
rejects them. If you are porting one, implement `transform_items(items)` instead.
The check is deliberately structural rather than a version flag, so a subclass
that simply implements the right method is recognised without declaring anything.

## 4. `run_job_unified` drives the work

`run_job_unified` is where the framework's contribution lives. Around your one
method it provides:

- **batching** — items are grouped rather than sent one at a time
- **bounded concurrency** — `--max-concurrency` in-flight requests, no more
- **retries** — tenacity-backed backoff on transient failures
- **checkpointing** — periodic flushes so a crash resumes instead of restarting
- **sharding** — `--num-shards` splits the input, `--shard-mode` decides how
- **data backend selection** — pandas, polars or Ray, with a matching runner

See [Checkpointing and resuming](../guides/checkpointing.md) and
[Sharding and concurrency](../guides/sharding-concurrency.md).

## 5. Your method runs

The only method a subclass must implement:

```python
async def transform_items(self, items: list[Any]) -> list[Any]:
    ...
```

The contract is narrow and worth stating precisely: **same order, same length**.
One result per input item, positionally aligned. Everything else — which items
you get, when, how often results are persisted — is the framework's concern.

`SwarmJob` also provides `transform_streaming`, which is the default path built on
top of `transform_items`, so implementing the latter is enough. Override
`transform_streaming` only when you need control over how items are consumed.

Available on `self`:

- `self.client` — an `AsyncOpenAI` already pointed at the swarm endpoint
- `self.model` — the model being served
- `self.kwargs` — the configured `request_params`: the provider parameters
  forwarded on every request, however they reached the job (`--job-kwargs`'s
  `request_params` key, or a `request_params=...` constructor argument)
- `self.output_cols` — the column(s) your results populate

## 6. Results are joined and written

Results are matched back to their input rows by id — the column named by
`--id-column`, or a generated one — and written to `--output`. With a directory
output and sharding, one Parquet file per shard may be written instead of a single
file.

Because the join is by id rather than position, a resumed run can write rows it
computed in an earlier attempt without recomputing them.

## Why the contract is shaped this way

The framework needs to reorder, batch, retry and persist your work. It can only
do that if your method is a **pure transform**: no I/O of its own, no assumptions
about which items arrive together, no side effects that a retry would duplicate.

That is the whole reason `transform_items` takes a list and returns a list, and
why checkpointing lives entirely outside it.
