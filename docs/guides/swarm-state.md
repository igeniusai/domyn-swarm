# Managing swarm state

Swarms outlive the process that created them. That is what makes
`domyn-swarm job submit --name my-swarm` possible hours after `up` returned, and
it is why there is a database to manage.

## What is stored and where

State lives in a SQLite database at `<DOMYN_SWARM_HOME>/swarm.db`, which defaults
to `~/.domyn_swarm/swarm.db`. Move it by setting `DOMYN_SWARM_HOME` — see
[Environment variables](../reference/environment.md).

Each record holds the deployment name, the configuration the swarm was created
from, platform identifiers such as job IDs and node assignments, and the endpoint
URL. `up` creates or updates the record; `down` and `db prune` remove records.

This is separate from `watchdog.db`, which is per swarm and holds replica health.
See [Watchdog and collector](../concepts/watchdog-collector.md).

## Automatic upgrades

You will not usually run a migration by hand. The CLI callback calls
`ensure_db_up_to_date` before any command that touches state — it is idempotent
and guarded by a process-local flag, so it costs nothing after the first call.

Three commands skip it deliberately, because they have no business paying for a
migration: `db`, `init` and `version`.

When a migration does run you will see it reported, since it is invoked with
`noisy=True`. If a release changes the schema, the first state-touching command
after upgrading absorbs the change.

## `db upgrade`

```bash
domyn-swarm db upgrade
```

Applies pending Alembic migrations to `swarm.db`. Mostly redundant given the
automatic upgrade, but useful to run the migration deliberately — before a batch
of automation, or to see the output on its own rather than mixed into another
command's.

## `db stamp`

```bash
domyn-swarm db stamp
```

Marks the database as being at the head revision **without running any
migrations**.

:::{warning}
`stamp` tells Alembic "this schema is already current" and is believed. Run it on
a database whose schema is *not* actually current and every future migration
starts from a false premise — later upgrades will skip the steps that would have
fixed it, and the failures surface far from the cause.

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

A record whose status probe **raises** is also treated as prunable — the reasoning
being that a swarm that cannot even be interrogated is not usable. That is
usually right, and it is worth knowing it is not the same test as "confirmed
dead": a transient failure to reach the platform during a prune can remove a
record for a swarm that is still running. Prune when the platform is reachable,
and prefer the prompt over `--yes` unless scripted.

Records with no resolvable deployment name are skipped rather than deleted. If
nothing qualifies you get `No dirty swarm records found.`

## Inspecting swarms

```bash
domyn-swarm swarm list             # probes live status (default)
domyn-swarm swarm list --no-probe  # cached info only, much faster
```

`list` renders a compact table: name, backend, phase, endpoint and notes. Probing
contacts each swarm's platform and load balancer, so with many swarms it is
noticeably slow — `--no-probe` shows what the state database already knows.

```bash
domyn-swarm swarm describe my-swarm
domyn-swarm swarm describe my-swarm -o yaml
domyn-swarm swarm describe my-swarm -o json
```

`describe` shows one swarm in detail **from local state, with no live probing** —
including its full resolved configuration, which is the practical way to recover
the config a running swarm was created with. `-o yaml` or `-o json` makes it
machine-readable.

For live health rather than recorded state, use `domyn-swarm status`. See
[Monitoring and troubleshooting](monitoring.md).

## Job records

Jobs are tracked too, which is what makes detached submission usable:

```bash
domyn-swarm job list                     # jobs and their statuses
domyn-swarm job status <job-id>           # one job
domyn-swarm job status <job-id> --refresh # re-probe the backend first
domyn-swarm job wait <job-id>             # block until it finishes
domyn-swarm job cancel <job-id>           # stop it
```

`--refresh` matters: without it you see the last recorded status, which may be
stale for a job whose process died without updating its record. With it, the
compute backend is probed and the record reconciled.

## When a record outlives its jobs

A swarm whose Slurm jobs are gone — cancelled outside domyn-swarm, or lost to a
node failure — leaves a record that `swarm list` still shows.

The order that works:

1. `domyn-swarm swarm list` to confirm the phase looks wrong
2. `domyn-swarm down <name>` to clear both the platform resources and the record
3. `domyn-swarm db prune` if `down` cannot resolve the swarm at all

`down` is the right first move because it removes both sides. `prune` is the
fallback for records too broken for `down` to act on.
