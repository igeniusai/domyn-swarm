# Quickstart

Stand up a vLLM endpoint on Slurm, run a batch job against it, and tear it down.

Everything below assumes an [installed](installation.md) domyn-swarm and access
to a Slurm cluster. For DGX Cloud Lepton, the same commands apply — only the
config's `backend` section changes, as described in
[Running on Lepton](../guides/lepton.md).

## 1. Prepare a YAML config

```yaml
# config.yaml
model: "HuggingFaceTB/SmolLM3-3B-Base"
name: smollm3
gpus_per_replica: 4
replicas: 4
backend:
  type: slurm
  partition: partition_name    # your HPC partition
  account: account_name        # your HPC account
  qos: qos_name                # qos for cluster + jobs
```

:::{note}
`model` may be a Hugging Face ID or a local path. If it is a Hugging Face ID,
make sure it has already been downloaded into your `HF_HOME` — the replicas run
offline.
:::

Every other field has a default. See [Configuration](../reference/configuration.md)
for the full set, or run `domyn-swarm init defaults` to record cluster-wide
defaults once instead of repeating them in every config.

## 2. Launch a swarm

```bash
domyn-swarm up -c config.yaml
```

This submits:

- an **array job** with 2 cluster replicas running vLLM servers
- a **load-balancer job** running Nginx, which waits on all replicas
- a **SQLite state record** for the swarm, created or updated locally

The state record is what lets later commands find the swarm by name. `up`
returns once the endpoint answers health checks.

## 3. Submit a typed job

Jobs take a DataFrame in and write a DataFrame out:

```bash
domyn-swarm job submit \
  --name my-swarm-name \
  --job-kwargs '{"temperature":0.3}' \
  --checkpoint-interval 16 \
  --max-concurrency 8 \
  --retries 2 \
  --input examples/data/chat_completion.parquet \
  --output results.parquet
```

Under the hood this spawns a driver that:

- reads `ENDPOINT=http://<endpoint-node>:9000`
- runs `python -m domyn_swarm.jobs.cli.run ...` via `srun` on Slurm, or the
  platform equivalent elsewhere
- streams prompts to answers with batching, backoff retries and checkpointing

With no job class given, the default is
`domyn_swarm.jobs:ChatCompletionJob`. See
[Submitting jobs](../guides/submitting-jobs.md) for the full flag set.

## 4. Submit a free-form Python script

When a job class is the wrong shape, send a script instead:

```bash
domyn-swarm job submit-script \
  --name my-swarm-name \
  examples/my_custom_driver.py -- --verbose --foo bar
```

Arguments after `--` are passed to the script rather than to domyn-swarm.

## 5. Check status

```bash
domyn-swarm status my-swarm-name
```

Reports the load-balancer endpoint alongside per-replica health — running,
unhealthy or failed — read from the watchdog database. See
[Monitoring and troubleshooting](../guides/monitoring.md) when a replica does not
come up.

## 6. Shut down

```bash
domyn-swarm down my-swarm-name
```

Stops the load balancer and all replica jobs.

## 7. List available swarms

```bash
domyn-swarm swarm list
```

## Next steps

- [Submitting jobs](../guides/submitting-jobs.md) — input formats, checkpointing, concurrency
- [Your first custom job](first-custom-job.md) — write your own `SwarmJob`
- [Running on Slurm](../guides/slurm.md) — Singularity images, bind mounts, modules
- [Configuration](../reference/configuration.md) — every config field
