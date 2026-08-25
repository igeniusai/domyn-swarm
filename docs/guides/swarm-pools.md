# Swarm pools

:::{note}
This guide is still being written. The feature it documents is itself incomplete, so there is little to describe yet.
:::

`domyn-swarm pool pool` validates a pool configuration and brings every swarm in it up, then immediately tears them all down again — there is no way to submit work to a pool yet, and the command's own help text is marked "not yet implemented". Do not build a workflow on it. To run several swarms today, launch each independently with `domyn-swarm up -c <config>` and address them by `--name`.
