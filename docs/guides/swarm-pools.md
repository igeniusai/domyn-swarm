# Swarm pools

:::{warning}
Swarm pools are **incomplete**. `domyn-swarm pool pool` validates the pool
configuration and brings every swarm in it up, then immediately tears them all
down again — there is no way to submit work to a pool. The command's own help text
is marked "not yet implemented".

Do not build a workflow on this. [Running several swarms today](#running-several-swarms-today)
describes what actually works.
:::

This page documents the configuration shape so it is ready when the feature
lands, and so nobody loses an afternoon discovering the above.

## Configuration shape

A pool is a list of swarms, each pointing at its own config file:

```yaml
# pool.yaml
pool:
  - name: qwen-32b
    config_path: configs/qwen3_32B.yaml
  - name: deepseek-r1
    config_path: configs/deepseek_r1.yaml
```

`SwarmPoolConfig.pool` is a list of `SwarmPoolElement`, each with:

- `name` — a label identifying this swarm within the pool
- `config_path` — path to a `DomynLLMSwarmConfig` YAML, read with
  `DomynLLMSwarmConfig.read`

The referenced files are ordinary swarm configs, exactly as used with
`domyn-swarm up -c`. Nothing about them is pool-specific, so the same config can
be used both ways.

## What the command does today

```bash
domyn-swarm pool pool pool.yaml
```

Note the doubled word: the subcommand is registered as `pool` inside the `pool`
app, so the invocation is `domyn-swarm pool pool`.

In order, it:

1. validates `pool.yaml` against `SwarmPoolConfig`
2. constructs one `DomynLLMSwarm` per element, reading each `config_path`
3. enters a pool context that brings every swarm up — then exits it immediately,
   tearing them all down

Step 3 is the problem: the body of the context is empty. Every swarm is allocated
and deallocated with no opportunity to submit anything.

What it *is* useful for is validating a set of configs — a malformed
`config_path` or an invalid swarm config fails before any resources are
requested. That is a real if narrow use.

## Running several swarms today

Launch each independently and address them by name:

```bash
domyn-swarm up -c configs/qwen3_32B.yaml
domyn-swarm up -c configs/deepseek_r1.yaml

domyn-swarm swarm list

domyn-swarm job submit --name qwen-32b   --input prompts.parquet --output qwen.parquet
domyn-swarm job submit --name deepseek-r1 --input prompts.parquet --output deepseek.parquet

domyn-swarm down qwen-32b
domyn-swarm down deepseek-r1
```

This is what the state database is for — every swarm is independently addressable
by name across processes, so a shell script or job array coordinates them without
needing pool support. See [Managing swarm state](swarm-state.md).

For A/B comparisons between checkpoints of the same model, `replicas` in a single
config launches multiple independent clusters, which may be what you wanted from
a pool in the first place. See [Configuration](../reference/configuration.md).
