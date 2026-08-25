# Configuration

Every runtime option lives in a single YAML file, loaded into
{py:class}`~domyn_swarm.config.swarm.DomynLLMSwarmConfig`. The tables below are
generated from the Pydantic models, so they always match the installed version.

A field marked **required** must be present. A field marked *computed* has a
default derived at load time from other fields — see the field's description for
the formula.

Values are resolved highest priority first:

1. the CLI arguments and this YAML config
2. `defaults.yaml`, written by `domyn-swarm init defaults`
3. the built-in defaults shown below

Any field you omit inherits the default, so configs can stay minimal.

## Models

```{include} ../_generated/config-tables.md
```
