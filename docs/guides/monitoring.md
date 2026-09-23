# Monitoring and troubleshooting

This page explains how to find health and runtime failures. For the health
reporting design, see [Watchdog and collector](../concepts/watchdog-collector.md).

For throughput, queue depth, and GPU utilization, see
[Metrics and dashboards](metrics.md).

## Reading `domyn-swarm status`

```bash
domyn-swarm status my-swarm-name
```

The output combines the serving phase and endpoint with per-replica health from
the swarm's `watchdog.db`.

Each replica row carries `replica_id`, `node`, `port`, `state`, `http_ready`,
`exit_code`, `exit_signal`, `fail_reason` and `last_seen`.

Read them in this order:

`state`
: the watchdog state: running, unhealthy, or failed

`http_ready`
: whether the replica answered its last HTTP probe. A replica can be *running*
  and not ready, which is the normal state of a large model still loading

`fail_reason`
: why the watchdog thinks it failed. The first thing to look at on a bad replica

`last_seen`
: when the watchdog last reported. A stale timestamp means the *watchdog* is gone,
  not the replica. The other fields are not reliable after the watchdog stops

Because this is observed health, a replica the platform believes is running will
still show as unhealthy here if it stopped answering probes. That disagreement is
information, not a bug.

## JSON output

```bash
domyn-swarm status my-swarm-name -o json
```

`--output` accepts `table` (default, a Rich view) or `json`. Anything else is
rejected outright.

The JSON schema is a public contract. It contains the serving phase, endpoint,
replica summary, per-replica rows, and an `errors` list.

## Job-level commands

These commands track jobs rather than replica health:

```bash
domyn-swarm job list                      # all jobs and their statuses
domyn-swarm job status <job-id>            # one job, from the record
domyn-swarm job status <job-id> --refresh  # re-probe the backend first
domyn-swarm job wait <job-id>              # block until it finishes
domyn-swarm job cancel <job-id>            # stop it
```

Use `--refresh` when a job status appears stale. Without it, the command reads
the last recorded value, which a process that died without updating its record
will never correct.

## When a replica is unhealthy

Use these steps in order:

1. Make sure that the model exists in `HF_HOME` on the compute nodes. Replicas
   run offline, so a missing Hugging Face model fails during load.
2. Read `fail_reason` in the replica row. It often names the failure.
3. Read the replica log under `backend.log_directory`. The default directory is
   `<home_directory>/logs`. The log contains the vLLM traceback.
4. Compare the restart count with `max_restarts`. After this limit, the watchdog
   stops retrying the replica.
5. Compare `readiness_timeout` with the actual model load time. The default is
   600 seconds. Increase it if the watchdog restarts a model that is still loading.

## When the endpoint never becomes ready

The load balancer waits for all replicas before it exposes the endpoint. One
stuck replica can block the swarm. Read each replica status first.

The `wait_endpoint_s` value, which defaults to 1200, limits the wait time for the
load-balancer script. `backend.endpoint.poll_interval` sets the probe interval.
If replicas are healthy and the endpoint is not, read the load-balancer log.

## When requests time out under load

If the endpoint is healthy but requests time out, examine concurrency.
`--max-concurrency` multiplied by `--num-shards` gives the in-flight request
count. A timeout for an idle endpoint can be too short when requests queue. See
[Sharding and concurrency](sharding-concurrency.md).

This is the case where health checks tell you least and metrics tell you most:
every replica can be running while the queue grows. If monitoring is enabled,
the vLLM queue-depth and throughput metrics show this state. See
[Metrics and dashboards](metrics.md).

## Log locations

| What | Where |
| --- | --- |
| Replica and load-balancer logs | `backend.log_directory`, default `<home_directory>/logs` |
| Replica health database | `watchdog.db`, per swarm |
| Swarm state database | `<DOMYN_SWARM_HOME>/swarm.db` |

For more detail from domyn-swarm itself, set `DOMYN_SWARM_LOG_LEVEL=DEBUG`.

## Getting more from the watchdog

If you run Ray and HTTP probes miss replica failures, enable
`watchdog.ray.enabled`. This option adds cluster liveness and capacity probes.
Set `ray.expected_tp` to the expected tensor-parallel world size to enforce
capacity. If it is unset, the watchdog probes only liveness.

The `WatchdogConfig` table lists probe intervals and thresholds. See
[Configuration](../reference/configuration.md).
