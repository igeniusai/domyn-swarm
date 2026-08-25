# Monitoring and troubleshooting

The operational counterpart to
[Watchdog and collector](../concepts/watchdog-collector.md), which explains why
health reporting is built this way. This page is for when something is wrong.

For *how fast* rather than *is it broken* — throughput, queue depth, GPU
utilisation — see [Metrics and dashboards](metrics.md).

## Reading `domyn-swarm status`

```bash
domyn-swarm status my-swarm-name
```

Two things are reported together: the **serving phase and endpoint**, from the
platform, and **per-replica health**, read from that swarm's `watchdog.db`.

Each replica row carries `replica_id`, `node`, `port`, `state`, `http_ready`,
`exit_code`, `exit_signal`, `fail_reason` and `last_seen`.

Read them in this order:

`state`
: the watchdog's verdict — running, unhealthy or failed

`http_ready`
: whether the replica answered its last HTTP probe. A replica can be *running*
  and not ready, which is the normal state of a large model still loading

`fail_reason`
: why the watchdog thinks it failed. The first thing to look at on a bad replica

`last_seen`
: when the watchdog last reported. A stale timestamp means the *watchdog* is gone,
  not the replica — a different problem, and one that makes the other fields
  untrustworthy

Because this is observed health, a replica the platform believes is running will
still show as unhealthy here if it stopped answering probes. That disagreement is
information, not a bug.

## JSON output

```bash
domyn-swarm status my-swarm-name -o json
```

`--output` accepts `table` (default, a Rich view) or `json`. Anything else is
rejected outright.

The JSON schema is treated as a **public contract** and changed with care, so it
is safe to build monitoring on. It carries the serving phase, the endpoint, the
replica summary and the per-replica rows, plus an `errors` list.

## Job-level commands

Separate from replica health — these track the work, not the endpoint:

```bash
domyn-swarm job list                      # all jobs and their statuses
domyn-swarm job status <job-id>            # one job, from the record
domyn-swarm job status <job-id> --refresh  # re-probe the backend first
domyn-swarm job wait <job-id>              # block until it finishes
domyn-swarm job cancel <job-id>            # stop it
```

Reach for `--refresh` when a job's status looks stuck. Without it you are reading
the last recorded value, which a process that died without updating its record
will never correct.

## When a replica is unhealthy

Ordered by how often each is the actual cause:

**1. The model is not in `HF_HOME`.** The most common Slurm failure by a wide
margin. Replicas run offline, so a Hugging Face ID that has not been downloaded
fails at load. Confirm the model is present and readable from the compute nodes,
not just from the login node.

**2. Read `fail_reason` on the replica row.** It is populated for a reason and
usually names the problem.

**3. Read the replica's log.** Under `backend.log_directory`, defaulting to
`<home_directory>/logs`. This is where a vLLM traceback actually lives —
`status` reports *that* it failed, the log says why.

**4. Check whether the watchdog gave up.** With `restart_policy` of `on-failure`
or `always`, a replica is restarted up to `max_restarts` times, with exponential
backoff between `restart_backoff_initial` and `restart_backoff_max`. Once
`max_restarts` is exhausted the replica is left failed and stops being retried,
so a swarm that looked recoverable earlier may simply be out of attempts.

**5. Compare `readiness_timeout` against real load time.** Default 600 seconds. A
large model on cold storage can exceed it, and the replica is then marked
unhealthy while still loading correctly — after which
`unhealthy_restart_after` restarts it, and it loads slowly again. A restart loop
with no error in the logs is this. Raise `readiness_timeout`.

## When the endpoint never becomes ready

The load balancer waits for **all** replicas before exposing the endpoint, so one
stuck replica holds up the whole swarm — which is why per-replica status is worth
reading before assuming the endpoint is at fault.

Relevant settings: `wait_endpoint_s` (default 1200) bounds how long the
load-balancer script waits, and `backend.endpoint.poll_interval` sets how often it
checks. If replicas are healthy and the endpoint is not, look at the
load-balancer job's own log rather than the replicas'.

## When requests time out under load

Endpoint healthy, individual requests failing: this is usually concurrency, not
health. `--max-concurrency` multiplied by `--num-threads` is the real in-flight
count, and a `--timeout` tuned against an idle endpoint will fire once requests
start queueing. See
[Sharding and concurrency](sharding-concurrency.md).

This is the case where health checks tell you least and metrics tell you most:
every replica is *running* and the queue is simply deeper than it can drain. If
monitoring is enabled, vLLM's queue-depth and throughput series show it directly
— see [Metrics and dashboards](metrics.md).

## Log locations

| What | Where |
| --- | --- |
| Replica and load-balancer logs | `backend.log_directory`, default `<home_directory>/logs` |
| Replica health database | `watchdog.db`, per swarm |
| Swarm state database | `<DOMYN_SWARM_HOME>/swarm.db` |

For more detail from domyn-swarm itself, set `DOMYN_SWARM_LOG_LEVEL=DEBUG`.

## Getting more from the watchdog

If replicas fail in ways the HTTP probe does not catch, and you run Ray, enable
`watchdog.ray.enabled` for cluster liveness and capacity checks alongside the
HTTP probe. Set `ray.expected_tp` to the expected tensor-parallel world size to
enable capacity enforcement; leaving it unset keeps liveness checking only.

Probe cadence and thresholds are all configurable — see the `WatchdogConfig`
table in [Configuration](../reference/configuration.md).
