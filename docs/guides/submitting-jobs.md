# Submitting jobs

`domyn-swarm job submit` runs a typed job — a DataFrame in, a DataFrame out —
against a running swarm. This page covers the choices that matter; the
[CLI reference](../reference/cli.md) carries every flag and its exact signature.

```bash
domyn-swarm job submit \
  my_module:CustomCompletionJob \
  --name my-swarm-name \
  --job-kwargs '{"temperature":0.2}' \
  --checkpoint-interval 16 \
  --input prompts.parquet \
  --output answers.parquet
```

## Choosing the job class

The positional argument is `<module>:<ClassName>`, naming a class that implements
`SwarmJob`. It defaults to `domyn_swarm.jobs:ChatCompletionJob`.

`--job-kwargs` takes a JSON object passed to the job's constructor, which is how
OpenAI parameters reach the client:

```bash
--job-kwargs '{"temperature":0.2,"top_p":0.9}'
```

Writing your own class: [Your first custom job](../getting-started/first-custom-job.md).

## Input and output paths

`--input` and `--output` take a Parquet file or a directory holding a Parquet
dataset, and both must be on a filesystem shared with the compute nodes.

Input additionally supports:

- **numeric brace ranges** — `input_00{0978..1955}.parquet`, and the
  `{0978-1955}` form, expanding to a file range
- **glob patterns** with the pandas backend — `data-*.parquet`

`--limit` / `-l` caps how many rows are read, which is what you want while
debugging against a small slice of a large dataset.

## Attaching to a swarm

Exactly one of:

- `--name` — attach to an existing swarm, found via the state record
- `--config` — start a fresh swarm from YAML for this job, then tear it down

Passing both is an error.

## Row identity

`--id-column` / `--id-col` names a column to use as a stable row id. It matters
more than it looks:

- without it, pandas uses the DataFrame index and polars generates a `_row_id`
- resume is then stable **only** if the input ordering and scan are identical
  across runs

For robust resume across restarts, or across different scan graphs, provide a
stable id column. See [Checkpointing and resuming](checkpointing.md).

## Checkpointing

Progress is written out periodically so a failed run can resume instead of
starting over. `--checkpoint-interval` sets the flush frequency in items,
`--checkpoint-tag` gives a run a stable identity to resume against, and
`--no-resume` and `--no-checkpointing` opt out in different ways.

Details, and the resume semantics that go with sharding:
[Checkpointing and resuming](checkpointing.md).

## Concurrency and sharding

`--max-concurrency` bounds in-flight requests to the endpoint. `--num-shards`
splits the input into shards. `--shard-mode` picks `id` for stable hashing or
`index` for legacy row-order sharding. `--retries` and `--timeout` govern
individual request failures.

These interact with resume, so read
[Sharding and concurrency](sharding-concurrency.md) before changing
`--num-shards` on a run you intend to resume.

## Choosing a data backend

`--data-backend` selects `pandas`, `polars` or `ray`, and `--runner` selects
`pandas` or `arrow` for the non-Ray backends. `--native-backend` enables native
execution and is **required for Ray**, with `--ray-address` pointing at the
cluster. `--native-batch-size`, `--backend-read-kwargs` and
`--backend-write-kwargs` tune the backend further.

Which to use, and why: [Choosing a data backend](data-backends.md).

## Output layout

With a directory output and the Polars runner, `--shard-output` writes one
Parquet file per shard, based on `--num-shards`, using the checkpoint outputs as
the source of truth.

## Running detached

`--detach` runs the job in a separate process and prints its PID, so the job
survives your terminal. Track it afterwards with `domyn-swarm job list`,
`job status`, `job wait` and `job cancel` — see
[Managing swarm state](swarm-state.md).

`--mail-user` enables email notification on completion where the compute backend
supports it.

## Submitting a plain script

When the job shape does not fit, send a script to the head node instead:

```bash
domyn-swarm job submit-script \
  --name my-swarm-name \
  path/to/script.py -- --foo 1 --bar 2
```

- the script file must exist
- exactly one of `--config` or `--name`, as above
- everything after `--` is forwarded to your script, not consumed by domyn-swarm

The script runs with `ENDPOINT` and `MODEL` already set in its environment, so it
can build its own client against the swarm.
