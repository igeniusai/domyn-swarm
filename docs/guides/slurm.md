# Running on Slurm

On Slurm, a swarm is a job array of vLLM replicas fronted by an Nginx load
balancer, with roles assigned by `SLURM_NODEID`.

## What `up` submits

```bash
domyn-swarm up -c config.yaml --replicas 3
```

- `-c/--config` — path to your YAML config
- `-r/--replicas` — override the replica count from the config

Three things happen:

1. an **array job** starts one vLLM server per replica
2. a **load-balancer job** starts Nginx, which waits until every replica is
   answering, then exposes a single endpoint
3. a **SQLite state record** is created or updated, which is what lets later
   commands address the swarm by name

## Building the Singularity images

domyn-swarm uses Singularity as its container engine on Slurm. Build the two
required images from the definition files in
[`examples/singularity_images/`](https://github.com/igeniusai/domyn-swarm/tree/main/examples/singularity_images),
on a machine with `sudo`, or with `--fakeroot` if your site enables it:

```bash
# NGINX load balancer image
sudo singularity build nginx.sif examples/singularity_images/nginx.def

# vLLM runtime image
sudo singularity build vllm.sif examples/singularity_images/vllm.def
```

Then point the config at the results:

```yaml
model: "deepseek-ai/DeepSeek-R1-0528"  # whatever model you want to deploy
image: /shared/images/vllm.sif         # vLLM container; optional if you run from a venv
backend:
  type: slurm
  endpoint:
    nginx_image: /shared/images/nginx.sif  # required for the load balancer
```

Practical notes:

- Put the `.sif` files on a shared path readable by every compute node.
- Singularity must be available on the execution nodes.
- If your site disables `--fakeroot`, build with admin privileges elsewhere and
  copy the `.sif` onto the shared filesystem.
- `image` is optional if you run vLLM from a virtual environment instead of a
  container; `endpoint.nginx_image` is not.

## Bind mounts

`backend.mounts` adds bind mounts to the vLLM containers:

```yaml
backend:
  type: slurm
  mounts:
    - /scratch/datasets                # bound at the same path inside the container
    - /host/config:/etc/app/config:ro  # host:container, with an option
```

Each entry is either `/path`, bound at the same path inside the container, or
`/host/path:/container/path` with an optional `:ro` or `:rw` suffix.

Entries are passed verbatim to Singularity's `--bind`. domyn-swarm validates only
the basic shape — an absolute source path, at most a `source:dest:opts` triple —
and the container runtime does the actual binding and reports its own errors, for
instance a missing host path or an invalid option. For the full bind syntax see
[Apptainer](https://apptainer.org/docs/user/main/bind_paths_and_mounts.html) or
[SingularityCE](https://docs.sylabs.io/guides/latest/user-guide/bind_paths_and_mounts.html).

## Modules and sbatch preamble

Two fields inject site-specific setup into the generated cluster script:

```yaml
backend:
  type: slurm
  modules:
    - cuda/12.1
    - singularity
  preamble:
    - "#SBATCH --exclusive"
    - "export NCCL_DEBUG=WARN"
```

`modules` become `module load` lines; `preamble` lines are inserted near the top
of the script, before the module loads, which makes them suitable for extra
sbatch directives or shell setup.

## Node selection and limits

`backend.partition`, `account` and `qos` are required. Beyond those,
`exclude_nodes` and `node_list` accept Slurm's own syntax
(`node[001-004]`), `time_limit` caps the allocation, and `mail_user` enables
END and FAIL notifications. The load-balancer job is configured separately under
`backend.endpoint`, including its own optional `qos` override.

Full field list: [Configuration](../reference/configuration.md).

## Shutting down

```bash
domyn-swarm down my-swarm-name
```

Takes a swarm name and stops the load balancer and every replica job via
`scancel`.
