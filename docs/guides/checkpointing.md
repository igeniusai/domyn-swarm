# Checkpointing and resuming

A wall-clock limit, a preempted node, or an unavailable endpoint can interrupt a
long batch run. Checkpointing lets the next attempt continue from completed work.

It is on by default.

## What gets written, and where

`--checkpoint-dir` sets the location, defaulting to `<swarm-dir>/checkpoints`.
`--checkpoint-tag` gives a run a stable identity across invocations. Without a
tag, a later run cannot identify earlier work from the same run.

Around the checkpoint file, four things appear on disk:

| Path | Purpose |
| --- | --- |
| `<name>.parquet` | the merged result of everything completed so far |
| `<name>.parquet.parts/` | one `part-<uuid>.parquet` per flush, not yet merged |
| `<name>.parquet.meta.json` | the input fingerprint, see below |
| `<name>.parquet.lock` | guards concurrent flushes |

Each flush creates a new part file instead of rewriting the result. A crash
during a flush cannot corrupt completed work. `finalize()` merges the parts into
the main file, keeps the last write for each row id, and deletes the parts.

Every write is atomic: content goes to a `.tmp` path and is moved into place with
`os.replace`, so a reader never sees a half-written file.

## How a resume decides what to skip

On startup, the manager reads the completed rows and every unmerged part. It
builds the set of finished row ids and removes them from the work list.

The filter uses the id, not the position. This behavior is why
[`--id-column`](submitting-jobs.md#row-identity) matters. Without a stable id
column, pandas uses the DataFrame index and polars generates `_row_id`. A resume
is correct only if the input order is identical between runs.

## The input fingerprint

This is the safety mechanism most likely to surprise you.

When a checkpoint is created, `.meta.json` stores a BLAKE3 fingerprint of the
input index and input column. A resume computes the fingerprint again. A
mismatch raises:

```text
Checkpoint input fingerprint does not match current data.
```

This error prevents results from joining the wrong input rows. If the input
changed, use `--no-resume` to recompute it or use a new `--checkpoint-tag`.

Three other consistency rules apply on resume:

- The manager rejects a checkpoint with duplicate index values.
- The manager rejects a checkpoint without the expected output columns. This
  often means that two jobs use the same tag.
- The manager drops checkpoint rows that are absent from the input and writes a
  warning. This permits a smaller input on resume.

## Flush frequency

`--checkpoint-interval` is the number of items per flush, default 16.

Each flush writes a Parquet file. A crash loses all work after the last flush.
Use a small interval for slow or expensive calls. A large interval is suitable
for fast calls.

## Turning it off

The two flags have different effects:

`--no-resume`
: Keep writing checkpoints, but ignore existing checkpoints. This option
  recomputes all results. Use it after an input change or a fix for incorrect
  output.

`--no-checkpointing`
: Do not write checkpoints. Use this option for short runs or inexpensive output
  that you can regenerate.

## Sharded runs and `--global-resume`

With `--num-shards > 1`, each shard uses a separate checkpoint directory. This
keeps one writer in each directory and permits concurrent flushes.

If `--num-shards` or `--limit` changes between runs, the shard assignment also
changes. Shard 5 can then receive a row that shard 3 completed. The checkpoint
for shard 3 is not visible to shard 5.

`--global-resume` filters the input against completed ids from all shards. Use it
after a deliberate change to the shard count or limit. Otherwise, keep
`--num-shards` fixed between resumed runs and leave this option off.

See [Sharding and concurrency](sharding-concurrency.md).

## Cloud storage

The shard store resolves its location through `fsspec`. A checkpoint directory
can therefore be an object-store URI:

```bash
--checkpoint-dir s3://bucket/checkpoints/run
```

Install the required filesystem package, such as `s3fs`, `gcsfs`, or `adlfs`.
If `fsspec` cannot resolve the URI, the error names the required package.

## A caution on concurrency

Each checkpoint directory must have one writer process. Each job has one driver
process, and each shard has a separate directory. Resumed runs skip completed
identifiers. A process-wide counter orders writes. Timestamps can sort files in
the wrong order.

Do not use the same checkpoint directory for two concurrent job invocations.
Give each invocation a different tag.
