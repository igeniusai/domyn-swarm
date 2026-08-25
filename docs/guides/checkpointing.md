# Checkpointing and resuming

A long batch run will be interrupted — a wall-clock limit, a preempted node, an
endpoint that stops answering. Checkpointing means the next attempt continues
rather than repaying for work already done.

It is on by default.

## What gets written, and where

`--checkpoint-dir` sets the location, defaulting to `<swarm-dir>/checkpoints`.
`--checkpoint-tag` gives a run a stable identity, which is what makes it
resumable across separate invocations — without a tag, a later run has no way to
recognise earlier work as its own.

Around the checkpoint file, four things appear on disk:

| Path | Purpose |
| --- | --- |
| `<name>.parquet` | the merged result of everything completed so far |
| `<name>.parquet.parts/` | one `part-<uuid>.parquet` per flush, not yet merged |
| `<name>.parquet.meta.json` | the input fingerprint, see below |
| `<name>.parquet.lock` | guards concurrent flushes |

Flushes go to a *new part file* rather than rewriting the result, so a crash
mid-flush cannot corrupt completed work. `finalize()` merges the parts into the
main file, deduplicating by row id and keeping the last write, then deletes them.

Every write is atomic: content goes to a `.tmp` path and is moved into place with
`os.replace`, so a reader never sees a half-written file.

## How a resume decides what to skip

On startup the manager reads the completed rows and every unmerged part, builds
the set of finished row ids, and filters them out of the work list.

That filtering is by **id, not position**, which is why
[`--id-column`](submitting-jobs.md#row-identity) matters. Without a stable id
column, pandas uses the DataFrame index and polars generates `_row_id`, and a
resume is only correct if the input ordering is identical between runs.

## The input fingerprint

This is the safety mechanism most likely to surprise you.

When a checkpoint is created, a blake3 fingerprint of the input — its index plus
the input column — is recorded in `.meta.json`. On resume the fingerprint is
recomputed and compared, and a mismatch raises:

```text
Checkpoint input fingerprint does not match current data.
```

That is deliberate and it is protecting you. The alternative is silently joining
new results onto rows they do not belong to. If you changed the input, the
checkpoint is not resumable: use `--no-resume` to recompute, or a fresh
`--checkpoint-tag`.

Three other consistency checks fire on resume:

- a checkpoint whose index contains duplicates is rejected outright
- a checkpoint missing the job's expected output columns is rejected — usually a
  sign the tag is being reused across two different jobs
- rows present in the checkpoint but absent from the input are dropped with a
  warning rather than an error, since a shrunken input is a normal thing to do

## Flush frequency

`--checkpoint-interval` is the number of items per flush, default 16.

The trade-off is direct: each flush costs a Parquet write, and everything since
the last flush is lost on a crash. Slow expensive calls justify a small interval;
fast cheap ones do not. If you find yourself raising it above a few hundred, the
run is probably fast enough not to need checkpointing at all.

## Turning it off

Two flags that sound similar and are not:

`--no-resume`
: keep writing checkpoints, but ignore any that exist. Forces recompute of
  everything. This is what you want after changing the input or fixing a bug that
  produced wrong output.

`--no-checkpointing`
: do not write checkpoints at all. Appropriate for short runs, or when the output
  is cheap to regenerate and you would rather not leave files behind.

## Sharded runs and `--global-resume`

With `--num-threads > 1`, each shard keeps its own checkpoint directory — that is
what makes concurrent flushing safe, since exactly one writer per directory is
assumed by construction.

The consequence: if `--num-threads` or `--limit` changes between runs, shard
assignment changes, and a row completed by shard 3 last time may be assigned to
shard 5 now, where its checkpoint is invisible.

`--global-resume` fixes that by filtering the input against the union of done ids
across *all* shards rather than per-shard. Use it when you have deliberately
changed the shard count or the limit. Otherwise keep `--num-threads` fixed
between resumed runs and leave it off.

See [Sharding and concurrency](sharding-concurrency.md).

## Cloud storage

The shard store resolves its location through `fsspec`, so a checkpoint directory
can be an object-store URI:

```bash
--checkpoint-dir s3://bucket/checkpoints/run
```

This needs the relevant filesystem extra installed — `s3fs`, `gcsfs`, `adlfs` —
and raises an `ImportError` naming them if `fsspec` cannot resolve the URI.

## A caution on concurrency

Exactly one writer process per checkpoint directory is assumed, and holds by
construction: one driver process per job, a distinct directory per shard, and
resumed runs skipping done ids. Ordering within a directory comes from a
process-wide counter, not from timestamps, because millisecond-resolution
filenames can sort in the opposite order of the writes.

It is **not** safe to point two separate job invocations at the same checkpoint
directory simultaneously. Give them different tags.
