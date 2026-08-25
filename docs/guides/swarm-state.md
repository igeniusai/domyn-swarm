# Managing swarm state

:::{note}
This guide is still being written. Until it lands, the commands are documented in the [CLI reference](../reference/cli.md).
:::

Swarm state lives in a local SQLite database, which is what lets `--name` and `swarm list` work across processes. Maintenance commands are `domyn-swarm db upgrade`, `db stamp` and `db prune`; inspection is `swarm list` and `swarm describe`.
