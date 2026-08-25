# Watchdog and collector

Replica health is not inferred from whether a process is running. domyn-swarm uses
a **watchdog** per replica and a single **collector** per swarm, and the split
exists for a specific reason worth understanding before changing this code.

## The watchdog supervises one replica

Each replica is launched via `domyn_swarm.runtime.watchdog`, which:

- spawns `vllm serve ...`
- probes HTTP `/health`, and optionally Ray, on an interval
- applies the restart policy — `always`, `on-failure` or `never` — plus
  `unhealthy_restart_after`, which forces a restart when a replica has been
  unhealthy for too long
- sends compact JSON status updates over **TCP** to the collector

Each update carries `state`, `http_ready`, `pid`, `exit_code`, `fail_reason`,
`agent_version` and `last_seen`.

A running process is not a healthy replica: vLLM can be alive and not serving,
particularly while loading a large model. That is why readiness is a probe and why
`readiness_timeout` exists separately from the restart policy.

All of it is configurable under `watchdog` — see
[Configuration](../reference/configuration.md).

## The collector owns the database

One collector runs per swarm, on the load-balancer node
(`domyn_swarm.runtime.collector`). It:

- listens on `--host` / `--port` for watchdog updates
- is the **only writer** to the per-swarm SQLite database, `watchdog.db`
- upserts into a `replica_status` table keyed by `(swarm_id, replica_id)`
- enables WAL and `busy_timeout` on a best-effort basis
- ignores malformed packets and transient SQLite errors rather than dying

Watchdogs find it via `--collector-address host:port`, which the Slurm backend
injects. You do not normally wire this by hand.

## Why a single writer

This is the design decision the split exists to make.

SQLite tolerates one writer at a time. With every replica writing its own status
directly, a large swarm would produce constant write contention on a shared
filesystem — exactly the conditions where SQLite locking behaves worst — and the
failures would be intermittent, load-dependent, and worst precisely when the swarm
is biggest and health information matters most.

Funnelling every write through one process removes the contention entirely
instead of trying to tune around it. The cost is one extra process and a TCP hop;
the benefit is that health reporting has no concurrency story at all.

It also explains the tolerance for bad input. A collector that died on a malformed
packet or a transient lock would take down health reporting for the whole swarm,
so it drops what it cannot parse and carries on. Losing one status update is
recoverable — the next probe supersedes it. Losing the collector is not.

## What reads it

`domyn-swarm status` reads `watchdog.db` to show per-replica health — running,
unhealthy or failed — with HTTP readiness and failure reasons, alongside the load
balancer endpoint.

So `status` reports **observed** health rather than asking the platform what it
thinks it scheduled. A replica Slurm believes is running will still show as
unhealthy here if it stopped answering probes.

Operational guidance: [Monitoring and troubleshooting](../guides/monitoring.md).

## Ray-aware checks

With `watchdog.ray.enabled`, the watchdog additionally probes Ray cluster liveness
and capacity, on top of the HTTP check. `ray.expected_tp` is the expected
tensor-parallel world size and enables the capacity check; leaving it unset
disables capacity enforcement while keeping liveness. `ray.status_grace_s` requires
Ray to report healthy for a window before the replica counts as ready, which
avoids flapping on a cluster that is still assembling.
