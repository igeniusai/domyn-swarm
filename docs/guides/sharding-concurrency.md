# Sharding and concurrency

:::{note}
This guide is still being written. Until it lands, the flags are documented in the [CLI reference](../reference/cli.md).
:::

Two independent dials: `--max-concurrency` bounds in-flight requests *within* a shard, while `--num-threads` splits the input into shards. `--shard-mode` accepts `id` for stable id hashing or `index` for legacy row-order sharding.
