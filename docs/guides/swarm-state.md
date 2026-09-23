# Managing swarm state

Swarms outlive the process that created them. That is what makes
`domyn-swarm job submit --name my-swarm` possible hours after `up` returned, and
it is why there is a database to manage.

## What is stored and where

State lives in a SQLite database at `<DOMYN_SWARM_HOME>/swarm.db`, which defaults
to `~/.domyn_swarm/swarm.db`. Set `DOMYN_SWARM_HOME` to move it. See
[Environment variables](../reference/environment.md).

Each record holds the deployment name and the configuration that created the
swarm. It also holds platform identifiers, node assignments, and the endpoint
URL. `up` creates or updates the record. `down` and `db prune` remove records.

This is separate from `watchdog.db`, which is per swarm and holds replica health.
See [Watchdog and collector](../concepts/watchdog-collector.md).

## Automatic upgrades

The CLI usually runs migrations for you. Its callback calls
`ensure_db_up_to_date` before any command that touches state. The operation is
idempotent and runs once per process.

Three commands skip the migration because they do not use state: `db`, `init`,
and `version`.

The CLI reports each migration because it invokes the operation with
`noisy=True`. After a schema change, the first command that uses state applies
the migration.

## `db upgrade`

```bash
domyn-swarm db upgrade
```

Applies pending Alembic migrations to `swarm.db`. Use this command before an
automated batch or when you need separate migration output. Other commands apply
the same migrations automatically.

## `db stamp`

```bash
domyn-swarm db stamp
```

Marks the database as being at the head revision without running migrations.

:::{warning}
Do not run `stamp` on a database with an outdated schema. Alembic will treat the
schema as current and skip required migrations. Later commands can then fail.

The legitimate use is narrow: a database whose schema is correct but which
predates Alembic having a revision record for it. If you are not sure that
describes your situation, run `db upgrade` instead.
:::

## `db prune`

```bash
domyn-swarm db prune          # prompts before deleting
domyn-swarm db prune --yes    # skip the prompt
```

Deletes records for swarms that are no longer alive. Each record is probed and
deleted when the serving phase is `FAILED`, `STOPPED` or `UNKNOWN`.

A raised status probe also marks a record as prunable. This does not prove that
the swarm stopped. A temporary platform failure can make `db prune` remove a
record for a running swarm. Run the command only when the platform is reachable.
Use the prompt unless automation requires `--yes`.

Records with no resolvable deployment name are skipped rather than deleted. If
nothing qualifies you get `No dirty swarm records found.`

## Inspecting swarms

```bash
domyn-swarm swarm list             # probes live status (default)
domyn-swarm swarm list --no-probe  # cached info only, much faster
```

`list` renders the name, backend, phase, endpoint, and notes in a compact table.
Probing contacts each swarm's platform and load balancer. It can take longer when
many swarms exist. `--no-probe` shows only the recorded state.

```bash
domyn-swarm swarm describe my-swarm
domyn-swarm swarm describe my-swarm -o yaml
domyn-swarm swarm describe my-swarm -o json
```

`describe` shows one swarm from local state without a live probe. The output
includes the resolved configuration that created the swarm. Use `-o yaml` or
`-o json` for machine-readable output.

For live health rather than recorded state, use `domyn-swarm status`. See
[Monitoring and troubleshooting](monitoring.md).

## Job records

The database also tracks jobs. This record supports detached submission:

```bash
domyn-swarm job list                     # jobs and their statuses
domyn-swarm job status <job-id>           # one job
domyn-swarm job status <job-id> --refresh # re-probe the backend first
domyn-swarm job wait <job-id>             # block until it finishes
domyn-swarm job cancel <job-id>           # stop it
```

Without `--refresh`, the command shows the last recorded status, which can be
stale for a job whose process died without updating its record. With it, the
compute backend is probed and the record reconciled.

## When a record outlives its jobs

A swarm record can remain after its Slurm jobs stop outside domyn-swarm. This can
happen after manual cancellation or a node failure.

The order that works:

1. Run `domyn-swarm swarm list` to make sure that the phase is wrong.
2. Run `domyn-swarm down <name>` to remove the platform resources and record.
3. If `down` cannot resolve the swarm, run `domyn-swarm db prune`.

`down` is the right first move because it removes both sides. `prune` is the
fallback for records too broken for `down` to act on.
