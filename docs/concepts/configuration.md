# Configuration precedence

A swarm value can come from four sources. The source determines which value the
swarm uses.

## The chain

Highest priority first:

1. CLI arguments and the YAML configuration for the swarm
2. `defaults.yaml`, written by `domyn-swarm init defaults`
3. Built-in defaults from the Pydantic models
4. Environment variables, which use separate precedence rules

Any field you omit inherits from the next level down. A configuration can
therefore contain only three or four lines.

## `domyn-swarm init defaults`

This command records the values that are the same for every swarm on your
cluster. You do not need to repeat these values in each configuration:

```bash
domyn-swarm init defaults
```

It prompts for Slurm partition, account and QoS, the endpoint's Nginx image and
port, the polling interval, and optionally the Lepton workspace and images. The
answers are written to `~/.domyn_swarm/defaults.yaml`, or to `-o/--output`.

Run the command again after the cluster configuration changes. The `--force`
flag overwrites an existing file. A swarm configuration can override individual
values such as `partition`, `qos`, or `nginx_image`.

## How defaults.yaml is found

The lookup, in order:

1. `Settings.defaults_file`, which the `DOMYN_SWARM_DEFAULTS` environment
   variable sets
2. otherwise the built-in candidate locations, of which
   `~/.domyn_swarm/defaults.yaml` is the usual one

The file is loaded once and cached for the process.

## "Computed" defaults that are actually required

The generated [Configuration reference](../reference/configuration.md) marks
some fields as computed.

Fields such as `backend.partition`, `backend.account` and `backend.qos` are shown
as computed. Their default is a factory that reads `defaults.yaml`. The factory
has no fallback. It raises this error if the key is empty or absent from both
`defaults.yaml` and the swarm YAML:

```text
Missing required configuration key: slurm.partition
```

Here, computed means that the field resolves from `defaults.yaml` or raises an
error. A field is optional only when its description gives a default or formula.

The same swarm configuration can behave differently with another
`defaults.yaml` file. The swarm configuration does not record this dependency.

## Environment variables are a different axis

`Settings` reads environment variables prefixed `DOMYN_SWARM_`, plus `.env` in the
working directory and `~/.domyn_swarm/.env`. Some fields carry explicit aliases
that drop the prefix, such as `VLLM_API_KEY`.

These are not swarm configuration fields, so they are not part of the chain
above. They configure the process, including state storage, log verbosity, and
API tokens. See [Environment variables](../reference/environment.md) for the full
list.

Two environment variables affect swarm configuration:

- `DOMYN_SWARM_HOME` supplies the default for `home_directory`. It changes
  where state, logs and checkpoints are written.
- `DOMYN_SWARM_DEFAULTS` selects the `defaults.yaml` file. It changes the second
  level of the precedence chain.

Both are read when the value is first resolved, so exporting them after a process
has started has no effect on it.

## Debugging where a value came from

Set `DOMYN_SWARM_LOG_LEVEL=DEBUG`. The defaults loader logs each resolved key and
the fallback that it considered. The log identifies values from the swarm YAML,
`defaults.yaml`, and built-in defaults.
