
# domyn-swarm

A simple CLI for launching and managing Slurm-backed LLM clusters (“swarms”) with optional replicas, load-balancing, and two high-level submission modes:

* **script** – run any arbitrary Python file on the head node
* **job** – run a strongly-typed `SwarmJob` (DataFrame → DataFrame) with built-in batching, retries and checkpointing

## Installation

`pip install git+ssh://git@github.com/igeniusai/domyn-swarm.git`

or, if using `uv`:

if you just want to install the package:

`uv pip install git+ssh://git@github.com/igeniusai/domyn-swarm.git`

if you want to add it as a dependency:

`uv add git+ssh://git@github.com/igeniusai/domyn-swarm.git`


---

## Quickstart

1. **Prepare a YAML config**
   Define your Slurm settings and model:

   ```yaml
   # config.yaml
   model: "mistralai/Mistral-7B-Instruct"
   nodes: 4
   gpus_per_node: 8
   cpus_per_task: 12
   mem_per_cpu: "11G"
   replicas: 2
   home_directory: ".domyn_swarm/"
   venv_path: ".venv"
   ```

   You can find more examples in the `examples/` folder

2. **Launch a fresh swarm**

```bash
   domyn-swarm up -c config.yaml
```

   This will:

   * submit an **array job** with 2 replicas of your cluster
   * submit a **load-balancer** job that waits on all replicas
   * print a `swarm_<jobid>.json` file containing the state related to configuration of the swarm

3. **Run a typed job on the cluster**
   The default class is `ChatCompletionJob` (`domyn_swarm.jobs:ChatCompletionJob`), which you can find at `src/domyn_swarm/jobs.py`

```bash
   domyn-swarm submit job \
    --state swarm_16803892.json \
    --job-kwargs '{"temperature":0.3,"batch_size":16,"parallel":8,"retries":2}' \
    --input examples/data/chat_completion.parquet \
    --output results.parquet
```

   Under the hood this:

   * reads `ENDPOINT=http://<lb-node>:9000`
   * in a single `srun` on the Load Balancer node, invokes `domyn_swarm.run_job`
   * streams prompts→answers in batches, retrying failures, checkpointing progress

4. **Run a free-form Python script**

```bash
   domyn-swarm submit script \
     --state swarm_16803892.json \
     examples/my_custom_driver.py -- --verbose --foo bar
```

   This runs:

```bash
   srun … python my_custom_driver.py --verbose --foo bar
```

   on the head node.

5. **Shut down your swarm**

```bash
   domyn-swarm down swarm_16803892.json
```

   Cancels both the LB job and the array job via `scancel`.

---

## Commands

```
Usage: domyn-swarm [OPTIONS] COMMAND [ARGS]…

Options:
  --install-completion    Install shell completion
  --show-completion       Show existing completion
  --help                  Show this help and exit

Commands:
  up        Launch a new swarm allocation
  status    Check existing swarm status (TBD)
  pool      Deploy multiple swarms from one config (TBD)
  down      Tear down a swarm (by state file)
  submit    Submit work (script | job) to a live swarm
```

### `domyn-swarm up`

Start a new allocation:

```bash
domyn-swarm up -c config.yaml \
  --replicas 3 \
  --reverse-proxy
```

* `-c/--config` — path to your YAML
* `-r/--replicas` — override number of replicas
* `--reverse-proxy` — (TBD) launch an Nginx running on the login node you're logged, so that you can access Ray dashboard via SSH tunneling

Produces `swarm_<jobid>.json` in your log directory.

### `domyn-swarm down`

```bash
domyn-swarm down logs/swarm_16803892.json
```

Stops the LB and all replica jobs via `scancel`.

### `domyn-swarm submit script`

Free-form script on the head node:

```bash
domyn-swarm submit script \
  --state logs/swarm_16803892.json \
  path/to/script.py -- --foo 1 --bar 2
```

* **script\_file**: your `.py` file (must exist)
* **--config** or **--state** (one only)
* **args…** after `--` are forwarded to your script

### `domyn-swarm submit job`

Typed DataFrame → DataFrame jobs:

```bash
domyn-swarm submit job \
  my_module:CustomCompletionJob \
  --state swarm_16803892.json \
  --job-kwargs '{"temperature":0.2,"batch_size":16}' \
  --input prompts.parquet \
  --output answers.parquet
```

* `<module>:<ClassName>` implementing `SwarmJob`
* **--job-kwargs** — JSON for the job’s constructor
* **--input** / **--output** — Parquet files on shared filesystem

Internally uses checkpointing, batching, and retry logic.

---

### Configuration: `DomynLLMSwarmConfig`

All runtime options for the swarm launcher live in a single YAML file that is loaded into the `DomynLLMSwarmConfig` dataclass.  
Below is an overview of every field, its purpose, and the default that will be used if you omit it.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| **model** | `str` | **required** | HF model ID or local path. Please note that this value will passed verbatim to `vllm serve`, thus it must be a valid path of HF model. If using an HF model, make sure it is available offline in the configured HF_HOME (hf_home in this configuration)|
| **hf_home** | `pathlib.Path` | `/leonardo_work/iGen_train/shared_hf_cache/` | HF cache dir mounted on workers. |
| **revision** | `str \| null` | `null` | Git tag/commit for the model (if using HF). |
| **nodes** | `int` | `4` | Number of **worker nodes** (one *vLLM* instance per node). |
| **gpus_per_node** | `int` | `4` | GPUs allocated on each worker. |
| **cpus_per_task** | `int` | `8` | vCPUs reserved per SLURM task. |
| **mem_per_cpu** | `str` | `"40G"` | Memory per CPU core (SLURM syntax). |
| **partition** | `str` | `"boost_usr_prod"` | SLURM partition to submit to. |
| **account** | `str` | `"iGen_train"` | SLURM account / charge code. |
| **vllm_image** | `str` | `/leonardo_work/iGen_train/fdambro1/images/vllm_0.9.0.1.sif` | Singularity image that runs the vLLM workers. |
| **nginx_image** | `str` | `/leonardo_work/iGen_train/fdambro1/images/nginx-dask.sif` | Image that runs the NGINX + Dask side-services. |
| **log_directory** | `pathlib.Path` | `./logs` | Directory where SLURM output/error logs are written. |
| **max_concurrent_requests** | `int` | `2000` | Upper bound enforced by vLLM REST gateway. |
| **shared_dir** | `pathlib.Path` | `/leonardo_work/iGen_train/shared` | Scratch area mounted on every node. |
| **poll_interval** | `int` | `10` | Seconds between `sacct` polling cycles while waiting for jobs. |
| **template_path** | `pathlib.Path` | *(auto-filled)* | Internal path of the Jinja2 SLURM script template; no need to touch. |
| **vllm_args** | `str` | `""` | Extra CLI flags passed verbatim to `python -m vllm.entrypoints.openai.api_server …`. |
| **vllm_port** | `int` | `8000` | Port where each worker’s OpenAI-compatible API listens. |
| **venv_path** | `pathlib.Path` | `./.venv` | Virtual-env used by the *driver* process (not the containers). |

---

## Troubleshooting

* **`JSONDecodeError` in `--job-kwargs`**
  Quote your JSON properly in the shell (e.g. `'{"foo":1,"bar":2}'`).

* **Nginx “permission denied”**
  We bind-mount a writable `cache/` dir into `/var/cache/nginx`. Ensure your log directory is writable.

* **Checkpoint files**
  Look under `.checkpoints/` in your working directory. Delete or rename to reset progress.

* **Model not found**
 If you get an error like this `("Error code: 404 - {'object': 'error', 'message': 'The model ``deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`` does not exist.', 'type': 'NotFoundError', 'param': None, 'code': 404}")`
 make sure you've downloaded the model locally so that vLLM can serve it. If the model is on HuggingFace, you can download it using `HF_HOME=$FAST/hf_shared_cache huggingface-cli download ORG/MODEL_NAME --repo-type model`, and you configure the hf_home value in the configuration file appropriately.

---

Happy swarming!
