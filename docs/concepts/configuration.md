# Configuration precedence

A value reaching a running swarm can come from four places. Knowing which one
won is the difference between a five-minute fix and an afternoon.

## The chain

Highest priority first:

1. **CLI arguments and the YAML config** — what you wrote for this swarm
2. **`defaults.yaml`** — written by `domyn-swarm init defaults`
3. **Built-in defaults** — the values in the Pydantic models
4. **Environment variables** — a separate axis; see below

Any field you omit inherits from the next level down, which is why configs can
stay to three or four lines.

## `domyn-swarm init defaults`

Records the values that are the same for every swarm on your cluster, so they stop
being repeated in every config:

```bash
domyn-swarm init defaults
```

It prompts for Slurm partition, account and QoS, the endpoint's Nginx image and
port, the polling interval, and optionally the Lepton workspace and images. The
answers are written to `~/.domyn_swarm/defaults.yaml`, or to `-o/--output`.

Re-run it whenever the cluster changes; `--force` overwrites an existing file.
Individual values can still be overridden per swarm — switch the `partition`,
`qos` or `nginx_image` for one deployment while everything else keeps the
recorded defaults.

## How defaults.yaml is found

The lookup, in order:

1. `Settings.defaults_file`, which the `DOMYN_SWARM_DEFAULTS` environment
   variable sets
2. otherwise the built-in candidate locations, of which
   `~/.domyn_swarm/defaults.yaml` is the usual one

The file is loaded once and cached for the process.

## "Computed" defaults that are actually required

This is the part that surprises people, and it is visible in the generated
[Configuration reference](../reference/configuration.md).

Fields like `backend.partition`, `backend.account` and `backend.qos` are shown as
*computed* rather than **required**, because their default is a factory that reads
`defaults.yaml`. But that factory has no fallback: if the key is missing or empty
in `defaults.yaml`, and you did not supply it in your YAML, it raises

```text
Missing required configuration key: slurm.partition
```

So *computed* means "resolved from `defaults.yaml`, or an error" — not "safe to
omit". A field is genuinely safe to omit only when its description names a
concrete default or a formula.

This is also why the same config file works for one colleague and fails for
another: their `defaults.yaml` differs, and nothing in the config file records
the dependency.

## Environment variables are a different axis

`Settings` reads environment variables prefixed `DOMYN_SWARM_`, plus `.env` in the
working directory and `~/.domyn_swarm/.env`. Some fields carry explicit aliases
that drop the prefix, such as `VLLM_API_KEY`.

These are not swarm config fields, and they do not sit in the chain above. They
configure the *process*: where state lives, log verbosity, API tokens. Full list:
[Environment variables](../reference/environment.md).

Two of them do reach into swarm configuration, which is worth knowing:

- **`DOMYN_SWARM_HOME`** supplies the default for `home_directory`, so it moves
  where state, logs and checkpoints are written
- **`DOMYN_SWARM_DEFAULTS`** selects which `defaults.yaml` is read, and therefore
  changes level 2 of the chain wholesale

Both are read when the value is first resolved, so exporting them after a process
has started has no effect on it.

## Debugging where a value came from

Set `DOMYN_SWARM_LOG_LEVEL=DEBUG`. The defaults loader logs each key it resolves
and the fallback it considered, which is usually enough to see whether a value
came from your YAML, from `defaults.yaml`, or from a built-in default.
