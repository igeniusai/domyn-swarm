# Watchdog and collector

Process state does not prove replica health. domyn-swarm uses one watchdog per
replica and one collector per swarm. This design prevents concurrent database
writes from the replicas.

## The watchdog supervises one replica

Each replica is launched via `domyn_swarm.runtime.watchdog`, which:

- It spawns `vllm serve ...`.
- It probes HTTP `/health`, and optionally Ray, on an interval.
- It applies the `always`, `on-failure`, or `never` restart policy.
- It uses `unhealthy_restart_after` to restart a replica after a long unhealthy
  period.
- It sends compact JSON status updates over TCP to the collector.

Each update carries `state`, `http_ready`, `pid`, `exit_code`, `fail_reason`,
`agent_version` and `last_seen`.

A running vLLM process can still be unable to serve requests while it loads a
large model. A separate readiness probe detects this state.
`readiness_timeout` limits how long the watchdog waits for readiness.

The `watchdog` section controls this behavior. See
[Configuration](../reference/configuration.md).

## The collector owns the database

One collector runs per swarm, on the load-balancer node
(`domyn_swarm.runtime.collector`). It:

- It listens on `--host` and `--port` for watchdog updates.
- It is the only writer to the per-swarm SQLite database, `watchdog.db`.
- It upserts into a `replica_status` table keyed by `(swarm_id, replica_id)`.
- It enables WAL and `busy_timeout` on a best-effort basis.
- It ignores malformed packets and transient SQLite errors instead of stopping.

Watchdogs find it via `--collector-address host:port`, which the Slurm backend
injects. You do not normally wire this by hand.

The port defaults to `9100` and is configurable as
`backend.endpoint.collector_port`. A `COLLECTOR_PORT` variable in the submission
environment overrides it for that swarm. The host is always the
load-balancer node, and replicas read both values from the swarm's
`serving/collector.env`.

A collector exits if another process owns its port. It writes
`collector: FATAL: cannot bind <host>:<port>: ...` to `logs/collector.log`.
The load-balancer job waits for the collector to create
`run/collector.ready` before it carries on, and fails with that log excerpt if
it never appears. Watchdogs tolerate a missing collector. Without the readiness
file, the swarm can serve traffic without reporting replica health.

## Why a single writer

This is the design decision the split exists to make.

SQLite supports one writer at a time. Direct writes from every replica cause
contention on a shared filesystem. The risk increases with the number of
replicas and can cause intermittent failures.

One collector serializes all writes and removes this contention. This design
adds one process and one TCP hop.

The collector drops malformed packets and continues after transient lock errors.
The next probe replaces a lost status update. This behavior keeps one bad update
from stopping health reporting for the swarm.

## What reads it

`domyn-swarm status` reads `watchdog.db`. It shows whether each replica is
running, unhealthy, or failed. It also shows HTTP readiness, failure reasons,
and the load-balancer endpoint.

`status` reports observed health rather than asking the platform what it
thinks it scheduled. A replica Slurm believes is running will still show as
unhealthy here if it stopped answering probes.

Operational guidance: [Monitoring and troubleshooting](../guides/monitoring.md).

## Ray-aware probes

With `watchdog.ray.enabled`, the watchdog also probes Ray cluster liveness and
capacity. `ray.expected_tp` sets the expected tensor-parallel world size and
enables capacity probes. If it is unset, the watchdog probes only liveness.
`ray.status_grace_s` requires a healthy Ray status for a set period before the
replica becomes ready.
