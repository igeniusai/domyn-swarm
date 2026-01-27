<p align="center">
  <picture>
      <source srcset="static/domyn-swarm-logo-white.svg" media="(prefers-color-scheme: dark)">
      <source srcset="static/domyn-swarm-logo-primary.svg" media="(prefers-color-scheme: light)">
      <img src="static/domyn-swarm-logo-primary.svg" alt="domyn-swarm" height="100">
   </picture>
</p>

> A simple, batteries‑included CLI and Python library for launching **LLM serving endpoints** and running **high‑throughput batch jobs** against them. First‑class support for **Slurm** (HPC) and **NVIDIA DGX Cloud Lepton**.

<p align="center">
<img src="https://github.com/igeniusai/domyn-swarm/actions/workflows/ci.yaml/badge.svg" alt="CI">
<img src="https://img.shields.io/badge/python-3.10%7C3.11%7C3.12%7C3.13-brightgreen?style=flat&logoColor=green" alt="Python">
<img src="https://img.shields.io/badge/License-Apache%202.0-blue" alt="License - Apache 2.0">
<img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff">
<img src="https://microsoft.github.io/pyright/img/pyright_badge.svg" alt="Pyright">
</p>

---

## Why Domyn‑Swarm?

Domyn‑Swarm gives you a **single, consistent workflow** to:

* stand up a **scalable LLM endpoint** (vLLM, OpenAI‑compatible),
* **submit jobs/scripts** that call that endpoint with **checkpointing, retries, and concurrency**, and
* **tear it all down** cleanly — across HPC (Slurm) and cloud (Lepton) backends.

It’s designed for **fast evaluation loops**, **robust batch inference**, and **easy backend extension**.

---

## Features

* **One CLI** for **up → job submit → status → down** across platforms
* **Serving/Compute backends** behind clean protocols → easy to add new targets (e.g., AzureML)
* **Health checks & readiness**:
  * Slurm: array replicas + LB, HTTP probe on `/v1/models`
  * Lepton: deployment state polling
* **SwarmJob API** (DataFrame in → DataFrame out)
  * Built‑in **batching**, **bounded concurrency**, **tenacity retries**, **checkpointing** (Parquet)
  * **Compat layer** for older job shape
  * **Pluggable data backends** for IO + execution (`pandas`, optional `polars`, optional `ray`)
* **Script runner** (submit any Python file to the compute backend)
* **State persistence** (using SQLite to store your swarms state)
* **Optional extras**: `pip install domyn-swarm[lepton]` (Lepton), `domyn-swarm[polars]`, `domyn-swarm[ray]`

---

## Supported backends

### Serving / compute backends

* **Slurm** (HPC) — uses singularity containers and job arrays to run model replicas
* **NVIDIA DGX Cloud Lepton** — Endpoint + Batch Job via Lepton SDK (optional extra)

### Data backends (job IO / execution)

* **pandas** — default (always installed)
* **polars** — optional extra, supports `scan_parquet`/streaming reads and directory outputs
* **ray** — optional extra, supports distributed execution via Ray Datasets (requires a Ray address)

---

## Installation

**PyPI (once published):**

> [!NOTE]
> We still haven't published the package on PyPI, but it will soon be available


```bash
pip install domyn-swarm
# Optional extras
pip install 'domyn-swarm[lepton]'
pip install 'domyn-swarm[polars]'
pip install 'domyn-swarm[ray]'
# or everything
pip install 'domyn-swarm[all]'
```


**From source (GitHub):**

```bash
RELEASE=v0.26.0
pip install git+ssh://git@github.com/igeniusai/domyn-swarm.git@$RELEASE
# or with uv
uv pip install git+ssh://git@github.com/igeniusai/domyn-swarm.git@$RELEASE # git+ssh://git@github.com/igeniusai/domyn-swarm.git[lepton]
uv add git+ssh://git@github.com/igeniusai/domyn-swarm.git@$RELEASE # Add --extra lepton
uv tool install --from git+ssh://git@github.com/igeniusai/domyn-swarm.git@$RELEASE --python 3.12 domyn-swarm
```

> Lepton users: install the extra and run `lep login` to initialize credentials.

---

## Quickstart

### 1) Prepare a YAML config

```yaml
# config.yaml
model: "HuggingFaceTB/SmolLM3-3B-Base"
gpus_per_replica: 16
replicas: 2
backend:
  type: slurm
  partition: partition_name    # your HPC partition
  account: account_name          # your HPC account
  qos: qos_name             # qos for cluster + jobs
```

> **Note**: `model` can be an HF ID or a local path. If HF, ensure it’s downloaded to your `HF_HOME`.

### 2) Launch a swarm

```bash
domyn-swarm up -c config.yaml
```

This submits:

* an **array job** with 2 cluster replicas (vLLM servers)
* a **load‑balancer** job (Nginx) that waits on all replicas
* updates/creates a local **SQLite** state record for the swarm

### 3) Submit a typed job (DataFrame → DataFrame)

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

Under the hood, this spawns a driver that:

* reads `ENDPOINT=http://<endpoint-node>:9000`
* runs `python -m domyn_swarm.jobs.cli.run ...` via `srun` (Slurm) or the platform equivalent
* streams prompts→answers with batching, backoff retries, checkpointing

### 4) Submit a free‑form Python script

```bash
domyn-swarm job submit-script \
  --name my-swarm-name \
  examples/my_custom_driver.py -- --verbose --foo bar
```

### 5) Check status (Slurm)

```bash
domyn-swarm status my-swarm-name
```

### 6) Shut down

```bash
domyn-swarm down my-swarm-name
```

### 7) List available swarms

```bash
domyn-swarm swarm list
```

---

## Lepton (DGX Cloud) configuration

Install the extra and login first:

```bash
lep login
```

Example configuration:

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

> Secrets: the endpoint can be token‑protected; Domyn‑Swarm stores the secret name in the serving handle and passes it to jobs as env (`DOMYN_SWARM_API_TOKEN` or secret ref).

---

## Commands

```
Usage: domyn-swarm [OPTIONS] COMMAND [ARGS]…

Options:
  --install-completion    Install shell completion
  --show-completion       Show existing completion
  --help                  Show this help and exit

Commands:
  version   Show the version of the domyn-swarm CLI
  up        Launch a swarm allocation with a configuration
  status    Check the status of the swarm allocation given its state file
  down      Shut down a swarm allocation
  job       Submit a workload to a Domyn-Swarm allocation.
  pool      Submit a pool of swarm allocations from a YAML config.
  init      Initialize a new Domyn-Swarm configuration.
  swarm     List existing swarms with a compact status view.
  db        Manage the Domyn-Swarm state database.
```

### `domyn-swarm up`

Start a new allocation:

```bash
domyn-swarm up -c config.yaml --replicas 3 --reverse-proxy
```

* `-c/--config` — path to your YAML config
* `-r/--replicas` — override number of replicas from config
* `--reverse-proxy/--no-reverse-proxy` — enable an optional reverse proxy for the allocation

### `domyn-swarm down`

```bash
domyn-swarm down my-beautiful-llm-swarm
```

Take a Swarm name as input. It stops the LB and all replica jobs via `scancel`.

### `domyn-swarm job submit`

Typed DataFrame → DataFrame jobs:

```bash
domyn-swarm job submit \
  my_module:CustomCompletionJob \
  --name my-swarm-name \
  --job-kwargs '{"temperature":0.2}' \
  --checkpoint-interval 16 \
  --input prompts.parquet \
  --output answers.parquet
```

* `<module>:<ClassName>` implementing `SwarmJob`, defaults to `domyn_swarm.jobs:ChatCompletionJob`
* **--input** / **--output** — Parquet file or directory (parquet dataset) on a shared filesystem.
  Input supports numeric brace ranges like `input_00{0978..1955}.parquet` (also `{0978-1955}`) to
  expand a file range, and pandas also supports wildcard glob patterns (e.g. `data-*.parquet`).
* **--job-kwargs** — JSON for the job’s constructor
* **--config** or **--name** (one only) — start a fresh swarm from YAML, or attach to an existing swarm
* **--checkpoint-dir** — where to store checkpoint state (defaults to `<swarm-dir>/checkpoints`)
* **--checkpoint-tag** — stable tag for checkpointing (useful to resume across runs)
* **--no-resume** — ignore existing checkpoints for this run (forces recompute)
* **--no-checkpointing** — disable checkpointing entirely
* **--checkpoint-interval** — flush interval for checkpointing (items per flush)
* **--max-concurrency** — concurrent in-flight requests
* **--retries** — retries for failed requests
* **--timeout** — per-request timeout in seconds
* **--num-threads** — shard count for non-ray execution (also used for directory shard outputs)
* **--limit** / **-l** - Limit the size to be read from the input dataset. Useful when debugging and testing to reduce the size of the dataset
* **--detach** - Detach the job from the current terminal, running in a different process (PID will be printed)
* **--mail-user** — enable job email notifications (when supported by the compute backend)
* **--data-backend** — Data backend for IO (`pandas`, `polars`, `ray`)
* **--runner** — Runner implementation for non-ray backends (`pandas`, `arrow`)
* **--shard-output** — When output is a directory and using the Polars runner, write one parquet
  file per shard (based on `--num-threads`) using checkpoint outputs as the source of truth.
* **--native-backend / --no-native-backend** — enable native backend execution (required for ray)
* **--native-batch-size** — batch size for native backend mode (optional; ray/polars use it)
* **--backend-read-kwargs** / **--backend-write-kwargs** — JSON dict forwarded to backend read/write
* **--id-column / --id-col** — Optional column name used for stable row ids
* **--ray-address** — Ray cluster address to connect to when using `--data-backend ray`


Internally uses checkpointing, batching, and retry logic.

Directory output (write a parquet dataset instead of materializing/concatenating on the driver):

```bash
domyn-swarm job submit \
  --name my-swarm-name \
  --input prompts.parquet \
  --output outputs/ \
  --num-threads 8
```

Polars scan example (uses `scan_parquet` under the hood):

```bash
domyn-swarm job submit \
  my_module:CustomCompletionJob \
  --name my-swarm-name \
  --input prompts.parquet \
  --output outputs/ \
  --data-backend polars \
  --runner arrow \
  --backend-read-kwargs '{"use_scan": true}'
```

In this mode, execution stays polars-native (batch iteration) and checkpoints are written as Arrow-backed
parquet shards (no pandas conversion). Using a directory `--output` streams output directly to disk.

Ray backend example (distributed execution via Ray Datasets):

```bash
domyn-swarm job submit \
  --name my-swarm-name \
  --input prompts.parquet \
  --output outputs_ray/ \
  --data-backend ray \
  --id-column request_id \
  --ray-address 'ray://<head-node>:10001'
```

> Ray requires a stable id column (`--id-column`) and an explicit Ray address (`--ray-address` or
> `DOMYN_SWARM_RAY_ADDRESS` / `RAY_ADDRESS`). Output paths are treated as directories by Ray.


### `domyn-swarm job submit-script`

Free-form script on the head node:

```bash
domyn-swarm job submit-script \
  --name my-swarm-name \
  path/to/script.py -- --foo 1 --bar 2
```

* **script\_file**: your `.py` file (must exist)
* **--config** or **--name** (one only)
* **args…** after `--` are forwarded to your script

### `domyn-swarm status`

Show a detailed, single‑swarm status view: phase, endpoint link, HTTP health, and backend diagnostics (e.g., Slurm replica/LB or Lepton raw state). Designed to use your terminal’s full width.

**Usage**

```bash
# Describe a specific swarm’s live status
domyn-swarm status <swarm-name>
```

**What it does**

* Resolves the swarm from the state DB
* Queries the serving backend for live status

> Tip: Status is about *runtime health*. For a static config overview use `domyn-swarm swarm describe` (when available) or inspect the state file.

---

### `domyn-swarm swarm list`

List all known swarms in a compact table with phase, endpoint, and quick notes. Great for an at‑a‑glance view across environments.

**Usage**

```bash
# Probe live status (default): shows HTTP and quick backend notes
domyn-swarm swarm list

# Faster: skip HTTP/LB probing and show only cached info
domyn-swarm swarm list --no-probe
```

**Columns**

* **Name** – swarm identifier from the state DB
* **Backend** – slurm / lepton (or other platforms)
* **Phase** – normalized serving phase (e.g., RUNNING, PENDING)
* **Endpoint** – URL (clickable in supported terminals)
* **Notes** – brief diagnostics (HTTP 200, `rep=` / `lb=` / `raw_state=`)

**Behavior**

* With `--probe` (default), each swarm is queried for fresh status (slightly slower)
* With `--no-probe`, output uses cached values only (fastest)

### `domyn-swarm swarm describe`

Show a detailed, static description of a swarm from the local state. No live status probing (use `domyn-swarm status` for that).

**Usage**

```bash
domyn-swarm swarm describe my-swarm-name
```

* **--output / -o** - Display the information in table format (default), YAML or JSON

### `domyn-swarm init defaults`

Speed up config authoring by pre‑filling common values with an interactive defaults file. Run:

```bash
domyn-swarm init defaults
```

You’ll be prompted for things like **Slurm** partition/account/QoS, endpoint **nginx image** and port, polling interval, and (optionally) **Lepton** workspace and images. The answers are saved to:

* `~/.domyn_swarm/defaults.yaml` (default), or a custom path with `-o/--output`.

At runtime, Domyn‑Swarm resolves settings with this precedence:

1. **CLI / YAML config** (highest)
2. **defaults.yaml** (from `init defaults`)
3. **Built‑in defaults** (lowest)

You can re‑run `init defaults` any time; use `--force` to overwrite an existing file. Individual values can still be overridden per‑swarm in your YAML (e.g., switch the `partition`, `qos`, or `nginx_image` for one deployment while keeping sensible defaults for the rest).

---

### Core Configuration: `DomynLLMSwarmConfig`

All runtime options for the swarm launcher live in a single YAML file that is loaded into the `DomynLLMSwarmConfig` dataclass.
Below is an overview of every field, its purpose, and the default that will be used if you omit it.

| Field                         | Type           | Default                                      | Purpose                                                                                                                                      |                                                                   |
| ----------------------------- | -------------- | -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| **model**                     | `str`          | **required**                                 | HF model ID or local path. Passed verbatim to `vllm serve`; must resolve to a local directory or an offline Hugging Face model in `HF_HOME`. |                                                                   |
| **revision**                  | `str          \| null`                                       | `null`                                                                                                                                       | Git tag/commit for the model (if using HF).                       |
| **replicas**                  | `int`          | `1`                                          | How many *independent* vLLM clusters to launch (useful for A/B tests).                                                                       | |
| **gpus_per_replica**          | `int`          | `4`                                          | How many GPUs should the replica use? This parameter will also be used to set `--tensor-parallel-size`| |
| **gpus\_per\_node**           | `int`          | `4`                                          | GPUs allocated on each worker node. It depends on the platform: our Slurm infrastructure used 4 GPUs nodes, but yours may have 8.                                                                                                          |                                                                   |
| **nodes**                     | `int`          | `math.ceil(replicas / replicas_per_node)` or `math.ceil((replicas * gpus_per_replica) / gpus_per_node)` for multi-gpu multi-node clusters                                          | Worker nodes per replica (one vLLM server per node).                                                                                         |                                                                   |
| **cpus\_per\_task**           | `int`          | `32 // replicas_per_node` or `32` for multi-gpu multi-node clusters                                   | vCPUs reserved per SLURM task.                                                                                                               |
| **replicas\_per\_node**       | `int`          | `gpus_per_node // gpus_per_replica` if `gpus_per_replica` <= `gpus_per_node` else `None` | How many model instances can share the same node (you won't need to set this unless you want multiple replicas per GPU, e.g. 2 replicas for each gpu) |
| **image**               | `str          \| pathlib.Path`                               | `null`                                                                                   | Can be either the path to a Singularity Image to be used with a Slurm backend or a docker image, used with Lepton                                 |
| **args**                | `str`          | `""`                                         | Extra CLI flags passed verbatim to `python -m vllm.entrypoints.openai.api_server …`.                                                         |                                                                   |
| **port**                | `int`          | `8000`                                       | Port where each worker’s OpenAI-compatible API listens.                                                                                      |                                                                   |
| **home\_directory**           | `pathlib.Path` | value of `DOMYN_SWARM_HOME`                            | Root folder for swarm state (auto-generated inside CWD).                                                                                     |                                                                   |
| **env**                 |  `dict` | `null` | A yaml dict of key values that will be set as environment variables |
| **wait_endpoint_s** | `int` | 600 | How many seconds should the lb script wait for the endpoint to go up | |
| **backend**                    | `BackendConfig` | *see below*                                  | Backend specific configurations, either Slurm or Lepton                                                             |                                                                   |
| **watchdog** | `WatchdogConfig` | *see below* | Watchdog specific configurations to be used when monitoring the spawned vLLM instances | |

### Backend Configuration: `BackendConfig`

#### SlurmConfig

| Field                         | Type           | Default                                      | Purpose                                                                                                                                      |                                                                   |
| ----------------------------- | -------------- | -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| **type** | `Literal["slurm"]` | "slurm" | Type of backend |
| **partition**                 | `str`          | **required**                           | SLURM partition to submit to.                                                                                                                |                                                                   |
| **account**                   | `str`          | **required**                               | SLURM account / charge code.                                                                                                                 |                                                                   |
| **qos** | `str` | **required** | SLURM qos where the cluster and load balancer jobs will be submitted |
| **ray\_port**                 | `int`          | `6379`                                       | Port for Ray’s GCS/head node inside each replica.                                                                                            |                                                                   |
| **ray\_dashboard\_port**      | `int`          | `8265`                                       | Ray dashboard (optional).                                                                                                                    |                                                                   |
| **venv\_path**                | `pathlib.Path \| null`                                       | `null`                                                                                                                                       | Virtual-env used by the *driver* process (not the containers).    |
| **time\_limit**               | `str`          | `"36:00:00"`                                 | Overall SLURM wall-clock limit for the allocation.                                                                                           |                                                                   |
| **exclude\_nodes**            | `str          \| null`                                       | `null`                                                                                                                                       | Nodes to exclude, e.g. `"node[001-004]"` (pass-through to SLURM). |
| **node\_list**                | `str          \| null`                                       | `null`                                                                                                                                       | Explicit node list, e.g. `"node[005-008]"`.                       |
| **log\_directory**            | `pathlib.Path \| null`                                       | `<home_directory>/logs`                                                                                                                      | Where SLURM stdout/stderr files are written.                      |
| **template\_path**            | `pathlib.Path` | *(auto-filled)*                              | Internal path of the Jinja2 SLURM script template; no need to modify.                                                                        |                                                                   |
| **nginx\_template\_path**     | `pathlib.Path` | *(auto-filled)*                              | Jinja2 template for the NGINX config. | |
| **mail_user**                 | `str` | `null` | Send Slurm END,FAIL signal notification about the resources deployed by domyn-swarm (job and cluster) | myemail@gmail.com |
| **endpoint**                  | `SlurmEndpointConfig` | `null` | |
| **modules** | `list[str]` | `[]` | List of modules to be loaded by slurm | |
| **preamble** | `list[str]` | `[]` | Additional list of sbatch directives to be added to cluster sbatch script | |

#### SlurmEndpointConfig

| Field                  | Type  | Default      | Purpose                                                         |
| ---------------------- | ----- | ------------ | --------------------------------------------------------------- |
| **cpus\_per\_task**    | `int` | `32`          | vCPUs for the driver process (launches and monitors the swarm). |
| **mem**                | `str` | `"16GB"`     | Physical memory for the driver job.                             |
| **threads\_per\_core** | `int` | `1`          | SMT threads to request per physical core.                       |
| **wall\_time**         | `str` | `"24:00:00"` | SLURM time limit for the driver job.                            |
| **nginx\_image**       | `str \| pathlib.Path` | **required** | Path to a singularity image running NGINX as load balancer for the swarm.|
| **nginx_timeout**      | `str \| int` | "60s" | HTTP timeout for NGINX proxy requests to model replicas. |
| **port**           | `int` | `9000`       | External port exposed by the NGINX load balancer. |
| **poll_interval**  | `int` | `10` | Seconds between status checks while waiting for the load balancer to become ready. |


#### Using Singularity on Slurm

In the current implementation of domyn-swarm, we use Singularity as container engine
If your cluster uses Singularity, you can build the required images from the definition files under `examples/singularity_images/` and point your config to the resulting `.sif` paths.

**Build the images** (on a machine with `sudo` *or* using `--fakeroot` if enabled):

```bash
# NGINX load balancer image
sudo singularity build nginx.sif examples/singularity_images/nginx.def

# vLLM runtime image
sudo singularity build vllm.sif examples/singularity_images/vllm.def
```

**Reference the images in your YAML:**

```yaml
model: "deepseek-ai/DeepSeek-R1-0528"  # Whatever model you want to deploy
image: /shared/images/vllm.sif         # vLLM container used on Slurm (optional if you run venv)
backend:
  type: slurm
  endpoint:
    nginx_image: /shared/images/nginx.sif  # required for the LB
```

**Notes**

* Place `.sif` files on a shared, readable path for all compute nodes.
* Ensure Singularity is available on execution nodes.
* If your site disables `--fakeroot`, build with admin privileges on a workstation and copy the `.sif` to the shared filesystem.


#### LeptonConfig

| Field          | Type                   | Default                  | Purpose                                                                                   |
| -------------- | ---------------------- | ------------------------ | ----------------------------------------------------------------------------------------- |
| `type`         | `Literal["lepton"]`    | *required*               | Discriminator to select the Lepton backend.                                               |
| `workspace_id` | `str`                  | *required*               | Lepton workspace identifier used for deployments and jobs.                                |
| `endpoint`     | `LeptonEndpointConfig` | `LeptonEndpointConfig()` | Serving endpoint configuration (image, shape, mounts, env, token secret).                 |
| `job`          | `LeptonJobConfig`      | `LeptonJobConfig()`      | Batch job configuration (image, shape, mounts, env).                                      |
| `env`          | `dict[str, str]`       | `{}`                     | Additional global environment variables applied to both endpoint and jobs where relevant. |

---


#### LeptonEndpointConfig

| Field                           | Type                | Default                     | Purpose                                                                                                 |
| ------------------------------- | ------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------- |
| `image`                         | `str`               | `"vllm/vllm-openai:latest"` | Container image used to run the serving endpoint (vLLM‑compatible OpenAI server).                       |
| `allowed_dedicated_node_groups` | `list[str] \| None` | `None`                      | Optional constraint to one or more Lepton dedicated node groups for the endpoint.                       |
| `resource_shape`                | `str`               | `"gpu.8xh200"`              | Lepton resource shape for endpoint replicas (e.g., GPU type/count and memory).                          |
| `allowed_nodes`                 | `list[str]`         | `[]`                        | Optional list of specific nodes (within a node group) where the endpoint is allowed to run.             |
| `mounts`                        | `list[MountLike]`   | `_default_mounts()`         | Filesystem mounts injected into the endpoint container (validated to Lepton `Mount` if SDK is present). |
| `env`                           | `dict[str, str]`    | `{}`                        | Environment variables for the endpoint container.                                                       |
| `api_token_secret_name`         | `str \| None`       | `None`                      | Name of the Lepton secret that holds the API token exposed to jobs/clients.                             |

**Notes**

* A validator normalizes `mounts` to real Lepton `Mount` instances when the Lepton SDK is available; otherwise accepts dicts.

---

#### LeptonJobConfig

| Field                           | Type                | Default                          | Purpose                                                                                            |
| ------------------------------- | ------------------- | -------------------------------- | -------------------------------------------------------------------------------------------------- |
| `allowed_dedicated_node_groups` | `list[str] \| None` | `None`                           | Optional constraint to one or more Lepton dedicated node groups for batch jobs.                    |
| `image`                         | `str`               | `"igeniusai/domyn-swarm:latest"` | Container image used by Domyn‑Swarm driver jobs submitted to Lepton.                               |
| `resource_shape`                | `str`               | `"gpu.8xh200"`                   | Lepton resource shape for the batch job execution.                                                 |
| `allowed_nodes`                 | `list[str]`         | `[]`                             | Optional list of specific nodes where the job is allowed to run.                                   |
| `mounts`                        | `list[MountLike]`   | `_default_mounts()`              | Filesystem mounts injected into the job container (validated to Lepton `Mount` if SDK is present). |
| `env`                           | `dict[str, str]`    | `{}`                             | Environment variables for the job container.                                                       |

**Notes**

* Same mounts normalization behavior as the endpoint config.

---


**Terminology**

* **MountLike**: either a Lepton SDK `Mount` object (when SDK is installed) or a dict with the same fields; validated at runtime.
* **Resource shape**: Lepton preset describing accelerator type/count and other resources (e.g., `gpu.4xh200`).


#### WatchdogConfig

| Field                              | Type                                                 | Default        | Purpose                                                                                                                  |                                                                                                                                  |
| ---------------------------------- | ---------------------------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| `enabled`                 | bool                                                 | `true`         | Master switch to enable/disable the per-replica watchdog process.                                                        |                                                                                                                                  |
| `probe_interval`          | int (seconds)                                        | `30`           | Interval between watchdog HTTP/Ray health probes.                                                                        |                                                                                                                                  |
| `http_path`               | str                                                  | `"/health"`    | HTTP path probed on the vLLM REST server to determine readiness/health. A leading `/` is added automatically if missing. |                                                                                                                                  |
| `http_timeout`            | float (seconds)                                      | `2.0`          | Timeout for each HTTP health probe request.                                                                              |                                                                                                                                  |
| `readiness_timeout`       | int (seconds)                                        | `600`          | Maximum time allowed for the server to become ready before being considered unhealthy.                                   |                                                                                                                                  |
| `restart_policy`          | Literal[`"always"`, `"on-failure"`, `"never"`]       | `"on-failure"` | When the watchdog should restart the child process (vLLM).                                                               |                                                                                                                                  |
| `unhealthy_restart_after` | int (seconds)                                        | `120`          | If the replica stays UNHEALTHY for this long, the watchdog forces a restart (or exits, depending on policy).             |                                                                                                                                  |
| `max_restarts`            | int                                                  | `3`            | Maximum number of restart attempts before giving up and leaving the replica in FAILED state.                             |                                                                                                                                  |
| `restart_backoff_initial` | int (seconds)                                        | `5`            | Initial delay before the first restart attempt.                                                                          |                                                                                                                                  |
| `restart_backoff_max`     | int (seconds)                                        | `60`           | Upper bound for the exponential backoff between restart attempts.                                                        |                                                                                                                                  |
| `kill_grace_seconds`      | int (seconds)                                        | `10`           | Grace period after SIGTERM before the watchdog sends SIGKILL to the child.                                               |                                                                                                                                  |
| `log_level`               | Literal[`"debug"`, `"info"`, `"warning"`, `"error"`] | `"info"`       | Log verbosity for the watchdog process.                                                                                  |                                                                                                                                  |
| `ray.enabled`             | bool                                                 | `false`        | Enable Ray-aware health checks (cluster liveness/capacity) in addition to HTTP checks.                                   |                                                                                                                                  |
| `ray.expected_tp`         | int                                                  | null           | `null`                                                                                                                   | Expected tensor-parallel world size (total GPUs for vLLM). Used by the Ray capacity check; `null` disables capacity enforcement. |
| `ray.probe_timeout_s`     | float (seconds)                                      | `120.0`        | Timeout for each Ray health probe command (e.g. `ray status`, `ray list nodes`).                                         |                                                                                                                                  |
| `ray.status_grace_s`      | float (seconds)                                      | `10.0`         | Ray must report healthy for at least this window before the watchdog treats it as fully ready.                           |                                                                                                                                  |
| `ray.probe_interval_s`    | float (seconds)                                      | `30.0`         | Interval between Ray health probes when Ray checks are enabled.                                                          |                                                                                                                                  |


---

*Tip:* Any field omitted from your YAML file inherits the default above, so you can keep configuration files minimal—only override what you need.

*Tip:* You can set some defaults for your specific environment by running `domyn-swarm init defaults`

---

## Environment variables

There is a variety of environment variables that you can set and will be picked up by `domyn-swarm` automatically

**Overview**
`Settings` centralizes configuration sourced from environment variables and optional `.env` files. By default it reads from:

* `.env` in the current working directory
* `~/.domyn_swarm/.env`

Environment variables use the prefix `DOMYN_SWARM_` (case-insensitive) **unless an explicit alias is defined** (see the table below). Values are parsed and validated via Pydantic.

| Name                    | Type                     | Purpose                                                              |
| ----------------------- | ------------------------ | -------------------------------------------------------------------- |
| `DOMYN_SWARM_LOG_LEVEL` | string                   | Global logging level (e.g., `DEBUG`, `INFO`, `WARNING`).             |
| `DOMYN_SWARM_HOME`      | path                     | Home/state directory for domyn‑swarm files (defaults to `~/.domyn_swarm`). |
| `DOMYN_SWARM_DEFAULTS`  | path (optional)          | Path to YAML with overridable defaults used by the defaults loader (e.g. `~/.domyn_swarm/defaults.yaml`).  |
| `API_TOKEN`             | secret string (optional) | API token to authenticate with the vLLM‑compatible server.           |
| `DOMYN_SWARM_MAIL_USER` | string (optional)        | Email address for Slurm job notifications.                           |
| `LEPTONAI_API_TOKEN`    | secret string (optional) | Token for Lepton API authentication.                                 |
| `LEPTON_WORKSPACE_ID`   | string (optional)        | Default Lepton workspace ID used for deployments/jobs.               |

## Python API (Programmatic usage)

> [!NOTE]
> This API is in constant evolution and you can expect breaking changes up to the final stable release
>
> Migration note: legacy `transform(df)`-based jobs are no longer supported; implement
> `transform_items(items)` (or rely on `transform_streaming` provided by `SwarmJob`).

In the [examples](examples/) folder, you can see some examples of programmatic usage of `DomynLLMSwarm` by instantiating a custom implementation of SwarmJob and how to run it via CLI or in a custom script: [examples/scripts/custom_main.py](examples/scripts/custom_main.py).

### Define a custom job

```python
import random
from typing import Any, List, Tuple

import pandas as pd
from domyn_swarm.jobs import SwarmJob


class MyCustomSwarmJob(SwarmJob):
    """
    Example custom job using the new SwarmJob API.

    - Reads prompts from the `input_column_name` column (default: "messages")
    - Produces three output columns: completion, score, current_model
    - No checkpointing/I-O logic here: the runner handles that.
    """

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        model: str = "",
        input_column_name: str = "messages",
        # We'll override outputs to three columns
        output_cols: str | list[str] = "result",
        checkpoint_interval: int = 16,
        max_concurrency: int = 2,
        retries: int = 5,
        timeout: float = 600,
        **extra_kwargs: Any,
    ):
        # Initialize the base job (creates self.client, stores kwargs, etc.)
        super().__init__(
            endpoint=endpoint,
            model=model,
            input_column_name=input_column_name,
            output_cols=output_cols,
            checkpoint_interval=checkpoint_interval,
            max_concurrency=max_concurrency,
            retries=retries,
            timeout=timeout,
            **extra_kwargs,
        )
        # Our job returns 3 values per item
        self.output_cols = ["completion", "score", "current_model"]

    async def transform_items(self, items: list[Any]) -> list[tuple[str, float, str]]:
        """
        Pure transform: items -> results (same order, same length).
        Each item here is expected to be a prompt string.

        Returns:
            List of tuples: (completion_text, random_score, model_tag)
        """
        # You can pass OpenAI params via job kwargs (e.g., temperature)
        temperature = float(self.kwargs.get("temperature", 0.7))

        results: list[tuple[str, float, str]] = []

        # Note: The executor calls this for single items via `_call_unit`,
        # but we support lists to keep the contract general.
        for prompt in items:
            # Async OpenAI client already configured to hit the swarm endpoint
            resp = await self.client.completions.create(
                model=self.model,
                prompt=prompt,
                **self.kwargs,  # forward any extra OpenAI parameters
            )
            completion_text = resp.choices[0].text or ""
            results.append(
                (
                    completion_text,
                    random.random(),                   # demo score
                    f"{self.model}_{temperature}",     # demo tag
                )
            )

        return results
```

### Run a custom job via CLI

```shell
PYTHONPATH=. domyn-swarm job submit examples.scripts.custom_job:MyCustomJob \
   --config examples/configs/deepseek_r1_distill.yaml \
   --input examples/data/completion.parquet \
   --output results/output.parquet \
   --job-kwargs '{"temperature": 0.2}'
```

### Use a custom job in a script (create a cluster and submit the job manually)

```python
from pathlib import Path
from domyn_swarm import DomynLLMSwarm, DomynLLMSwarmConfig
from mypkg.jobs import MyCustomSwarmJob

cfg = DomynLLMSwarmConfig.read("config.yaml")

with DomynLLMSwarm(cfg=cfg) as swarm:
    job = MyCustomSwarmJob(endpoint=swarm.endpoint, model=swarm.model, max_concurrency=16, temperature=0.2)
    swarm.submit_job(job, input_path=Path("prompts.parquet"), output_path=Path("answers.parquet"))
```

---

## Architecture (in brief)

* **DomynLLMSwarm** (context manager) → brings endpoint up/down, submits work
* **Deployment** → pairs `ServingBackend` + `ComputeBackend`
* **Serving (Slurm/Lepton)** → `create_or_update`, `wait_ready`, `status`, `delete`
* **Compute (Slurm/Lepton)** → `submit`, `wait`, `cancel` (+ defaults helpers)
* **SwarmJob API** → implement `transform_items(items)`; batching/retries/checkpointing provided by the framework
* **State** → saved/loaded locally; rehydrate swarm for later submissions

---

## Watchdog & Collector

Domyn-Swarm uses a lightweight **watchdog + collector** pair to monitor vLLM replicas and persist their health.

- Each replica is launched via `domyn_swarm.runtime.watchdog`, which:
  - Spawns `vllm serve …`
  - Probes HTTP `/health` (and optionally Ray)
  - Applies restart policy (`always` / `on-failure` / `never`) plus `unhealthy_restart_after` for forced restarts
  - Sends compact JSON status updates (state, `http_ready`, `pid`, `exit_code`, `fail_reason`, `agent_version`, `last_seen`, …) over **TCP** to the collector.

- A single **collector** (`domyn_swarm.runtime.collector`) runs per swarm (on the LB node) and:
  - Listens on `--host` / `--port` for watchdog updates
  - Acts as the **only writer** to a per-swarm SQLite DB (`watchdog.db`)
  - Upserts into a `replica_status` table keyed by `(swarm_id, replica_id)`
  - Enables WAL / `busy_timeout` on a best-effort basis and ignores malformed packets or transient SQLite errors.

- Watchdogs discover the collector via `--collector-address host:port` (injected by the Slurm backend); you normally don’t have to wire this manually.

- `domyn-swarm status` reads from `watchdog.db` to show per-replica health (running/unhealthy/failed, HTTP readiness, and failure reasons) alongside the load balancer endpoint.
---

## Contributing

We welcome issues and PRs! Please see:

* `CONTRIBUTING.md` — how to propose changes, coding style, DCO/CLA (as applicable)

---

## License

Licensed under the **Apache License, Version 2.0**. See `LICENSE` and `NOTICE`.

---

## Acknowledgements

* Built on **vLLM** and **Ray** (Apache‑2.0)
* Optional **NVIDIA DGX Cloud Lepton** integration via the Lepton SDK

Happy swarming! 🚀
