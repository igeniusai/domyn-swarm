# The SwarmJob lifecycle

This page describes how `domyn-swarm job submit` produces a Parquet output file.
It also defines the work that a custom job must do.

## 1. The CLI resolves the class

`job submit` takes `<module>:<ClassName>` and imports it, defaulting to
`domyn_swarm.jobs:ChatCompletionJob`. `--job-kwargs` is parsed as JSON and passed
to the constructor as configuration overrides. Put provider parameters such as
`temperature` and `top_p` under `request_params`. The CLI passes these values to
the client without interpreting them:

```json
{"max_concurrency": 8, "request_params": {"temperature": 0.2}}
```

The swarm is located either by `--name`, rehydrated from the state record, or
created fresh from `--config`.

## 2. A driver process starts beside the endpoint

The compute backend submits a process with `ENDPOINT` and `MODEL` in its
environment. Slurm uses `srun`, and Lepton uses a batch job. The process runs
`python -m domyn_swarm.jobs.cli.run`.

The job does not execute inside the serving container. It is an HTTP client
like any other.

## 3. The API version is resolved

`resolve_job_api` decides which execution path applies, in this order:

1. A class-level `api_version >= 2` means the current API.
2. Otherwise, an override of `transform_items` or `transform_streaming` means the
   current API.
3. Otherwise, an override of `transform` or `run` means the legacy API.
4. All other classes use the legacy API.

`_ensure_new_api` rejects legacy `transform(df)` jobs. To port one, implement
`transform_items(items)`. Method-based detection accepts a subclass that
implements the current method without an explicit version flag.

## 4. `run_job_unified` drives the work

`run_job_unified` provides these services around the job method:

- Batching groups items instead of sending one item at a time.
- Bounded concurrency limits in-flight requests with `--max-concurrency`.
- Retries use Tenacity backoff for transient failures.
- Checkpointing writes periodic results for resume after a failure.
- Sharding splits input with `--num-shards` and `--shard-mode`.
- Data backend selection chooses pandas, polars, or Ray with a matching runner.

See [Checkpointing and resuming](../guides/checkpointing.md) and
[Sharding and concurrency](../guides/sharding-concurrency.md).

## 5. Your method runs

A subclass must implement one method:

```python
async def transform_items(self, items: list[Any]) -> list[Any]:
    ...
```

The method must return one result for each input item, in the same order. The
framework controls item selection, scheduling, and result persistence.

`SwarmJob` also provides `transform_streaming`, which is the default path built on
top of `transform_items`, so implementing the latter is enough. Override
`transform_streaming` only when you need control over how items are consumed.

The method can use these attributes:

- `self.client`: An `AsyncOpenAI` client for the swarm endpoint.
- `self.model`: The model that the swarm serves.
- `self.kwargs`: Provider parameters from `request_params`, whether supplied by
  `--job-kwargs` or the constructor.
- `self.output_cols`: The columns that receive the results.

## 6. Results are joined and written

The framework matches results to input rows by the `--id-column` value or a
generated identifier. It writes the joined data to `--output`. A sharded
directory output can contain one Parquet file per shard.

Because the join is by id rather than position, a resumed run can write rows it
computed in an earlier attempt without recomputing them.

## Why the contract is shaped this way

A pure transform performs no I/O and has no retry-sensitive side effects. It
also does not depend on item grouping. This contract lets the framework reorder,
batch, retry, and persist work.

For this reason, `transform_items` takes and returns a list. Checkpointing stays
outside the method.
