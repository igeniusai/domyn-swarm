# Running on Lepton

NVIDIA DGX Cloud Lepton is supported through the Lepton SDK: a serving endpoint
plus batch jobs, in place of Slurm's array and load-balancer jobs.

## Prerequisites

```bash
pip install 'domyn-swarm[lepton]'
lep login
```

`lep login` initialises the credentials domyn-swarm uses to create deployments
and jobs.

## Configuration

```yaml
backend:
  type: lepton
  workspace_id: workspace_id
  endpoint:
    image: vllm/vllm-openai
    allowed_dedicated_node_groups:
      - nodegroup-xx
    resource_shape: gpu.4xh200
    env:
      HF_HOME: /mnt/lepton-shared-fs/hf_home/
  job:
    allowed_dedicated_node_groups:
      - nodegroup-xx
    image: igeniusai/domyn-swarm:latest
```

The two sub-sections are deliberately separate: `endpoint` configures the
container that serves the model, `job` configures the containers that run your
batch work. They can use different images and different resource shapes — the
serving side needs GPUs for the model, the job side mostly needs enough CPU and
network to keep the endpoint busy.

`resource_shape` is a Lepton preset describing accelerator type and count, for
example `gpu.4xh200`. `mounts` on either section accepts Lepton `Mount` objects,
or plain dicts with the same fields when the SDK is not installed.

Full field list: [Configuration](../reference/configuration.md).

## Secrets and API tokens

The endpoint can be token-protected. domyn-swarm stores the *name* of the Lepton
secret in the serving handle — never the value — and passes it to jobs as
`DOMYN_SWARM_API_TOKEN` or as a secret reference:

```yaml
backend:
  type: lepton
  endpoint:
    api_token_secret_name: my-endpoint-token
```

Private registries are handled by `image_pull_secrets` on the endpoint or job
section.

## Differences from Slurm

Worth knowing before you port a Slurm config across:

- **No load-balancer job.** The Lepton endpoint fronts its own replicas, so
  there is no Nginx job and nothing corresponding to
  `backend.endpoint.nginx_image`.
- **Readiness is deployment-state polling**, not an HTTP probe on `/v1/health`.
  domyn-swarm asks Lepton whether the deployment is ready rather than testing the
  endpoint itself.
- **No Singularity.** Images are Docker images pulled by Lepton, so
  `examples/singularity_images/` is irrelevant here.
- **Node selection is by node group**, via `allowed_dedicated_node_groups` and
  optionally `allowed_nodes`, rather than Slurm partitions and node lists.

The CLI is otherwise identical: `up`, `job submit`, `status` and `down` behave
the same way, as in [Quickstart](../getting-started/quickstart.md).
