<p align="center">
  <picture>
      <source srcset="static/domyn-swarm-logo-white.svg" media="(prefers-color-scheme: dark)">
      <source srcset="static/domyn-swarm-logo-primary.svg" media="(prefers-color-scheme: light)">
      <img src="static/domyn-swarm-logo-primary.svg" alt="domyn-swarm" height="100">
   </picture>
</p>
<p align="center">
<img src="https://github.com/igeniusai/domyn-swarm/actions/workflows/ci.yaml/badge.svg" alt="CI">
<img src="https://img.shields.io/badge/python-3.10%7C3.11%7C3.12%7C3.13-brightgreen?style=flat&logoColor=green" alt="Python">
<img src="https://img.shields.io/badge/License-Apache%202.0-blue" alt="License - Apache 2.0">
<img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff">
<img src="https://microsoft.github.io/pyright/img/pyright_badge.svg" alt="Pyright">
</p>

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

or to install it globally:

`uv tool install --from git+ssh://git@github.com/igeniusai/domyn-swarm.git --python 3.12 domyn-swarm`

---

## Quickstart

1. **Prepare a YAML config**
   Define your Swarm settings and model:

   ```yaml
   # config.yaml
   model: "mistralai/Mistral-7B-Instruct"
   gpus_per_replica: 16
   replicas: 2
   backend:
      type: slurm
      partition: boost_usr_prod
      account: igen_train
      qos: qos_llm_min
   ```

> [!NOTE]
> `model` can be either an HF model or a path to a directory with a model compatible with vllm. If using an HF id, make sure that the model is saved locally in your `HF_HOME`

   You can find more examples in the [examples/configs](examples/configs) folder

2. **Launch a fresh swarm**

```bash
   domyn-swarm up -c config.yaml
```

   This will:

   * submit an **array job** with 2 replicas of your cluster
   * submit a **load-balancer** job that waits on all replicas
   * create a sqlite db with an entry containing the state related to the swarm you just created

3. **Run a typed job on the cluster**
   The default class is `ChatCompletionJob` (`domyn_swarm.jobs:ChatCompletionJob`), which you can find at [src/domyn_swarm/jobs.py](src/domyn_swarm/jobs.py)

```bash
   domyn-swarm job submit \
    --name my-swarm-name \
    --job-kwargs '{"temperature":0.3,"checkpoint_interval":16,"max_concurrency":8,"retries":2}' \
    --input examples/data/chat_completion.parquet \
    --output results.parquet
```

   Under the hood this:

   * reads `ENDPOINT=http://<lb-node>:9000`
   * in a single `srun` on the Load Balancer node, invokes `domyn_swarm.jobs.run`
   * streams prompts→answers in batches, retrying failures, checkpointing progress

4. **Run a free-form Python script**

```bash
   domyn-swarm job submit-script \
     --name my-swarm-name \
     examples/my_custom_driver.py -- --verbose --foo bar
```

   This runs:

```bash
   srun … python my_custom_driver.py --verbose --foo bar
```

   on the head node.

5. **Check the status of your cluster**

```bash
   domyn-swarm status swarm_16803892.json
```



5. **Shut down your swarm**

```bash
   domyn-swarm down swarm_16803892.json
```

   Cancels both the LB job and the array job via `scancel`.

### Lepton compatibility

Together with slurm, we currently support [NVIDIA DGX Cloud Lepton](https://www.nvidia.com/en-us/data-center/dgx-cloud-lepton/) as cloud infrastructure to deploy endpoints and workloads.
In order to use it, make sure you have installed domyn-swarm with the lepton extra

```yaml
backend:
  type: lepton
  workspace_id: j90afpem
  endpoint:
    image: vllm/vllm-openai
    allowed_dedicated_node_groups:
      - nv-domyn-nebius-h200-01-lznuhuob
    resource_shape: gpu.4xh200
    env:
      HF_HOME: /mnt/lepton-shared-fs/hf_home/
  job:
    allowed_dedicated_node_groups:
      - nv-domyn-nebius-h200-01-lznuhuob
    image: igeniusai/domyn-swarm:latest
```

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
  up        Launch a new swarm allocation
  status    Check the status of the swarm allocation
  down      Shut down a swarm allocation
  submit    Submit a workload to a Domyn-Swarm allocation.
  pool      Submit a pool of swarm allocations from a YAML config.
  init      Initialize a new Domyn-Swarm configuration.
```

### `domyn-swarm up`

Start a new allocation:

```bash
domyn-swarm up -c config.yaml \
  --name my-beautiful-llm-swarm \
  --replicas 3 \  # I'm overriding what's in the configuration file
  --reverse-proxy
```

* `-c/--config` — path to your YAML
* `-n/--name` - Name of the swarm allocation. If not provided, a random name will be generated.
* `-r/--replicas` — override number of replicas
* `--reverse-proxy` — (TBD) launch an Nginx running on the login node you're logged, so that you can access Ray dashboard via SSH tunneling


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
  --job-kwargs '{"temperature":0.2,"checkpoint_interval":16}' \
  --input prompts.parquet \
  --output answers.parquet
```

* `<module>:<ClassName>` implementing `SwarmJob`, defaults to `domyn_swarm.jobs:ChatCompletionJob`
* **--input** / **--output** — Parquet files on shared filesystem
* **--job-kwargs** — JSON for the job’s constructor
* **--config** or **--state** (one only)  -  the definition or state of the cluster where the job will be submitted
* **--checkpoint-interval** - batch size of the requests to be sent to be processed. Once a batch has finished processing, the checkpoint will be updated
* **--max-concurrency** - Number of concurrent requests to process
* **--retries** - Number of retries for failed requests
* **--num-threads** - How many threads should be used by the driver to run the job
* **--limit** / **-l** - Limit the size to be read from the input dataset. Useful when debugging and testing to reduce the size of the dataset
* **--detach** - Detach the job from the current terminal, running in a different process (PID will be printed)


Internally uses checkpointing, batching, and retry logic.


### `domyn-swarm job submit-script`

Free-form script on the head node:

```bash
domyn-swarm job submit-script \
  --name my-swarm-name \
  path/to/script.py -- --foo 1 --bar 2
```

* **script\_file**: your `.py` file (must exist)
* **--config** or **--jobid** (one only)
* **args…** after `--` are forwarded to your script

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
| **image**               | `str          \| pathlib.Path`                               | `null`                                                                                   | Can be either the path to a Singularity Image to be used with a Slurm backend or a docker image, used with                                 |
| **args**                | `str`          | `""`                                         | Extra CLI flags passed verbatim to `python -m vllm.entrypoints.openai.api_server …`.                                                         |                                                                   |
| **port**                | `int`          | `8000`                                       | Port where each worker’s OpenAI-compatible API listens.                                                                                      |                                                                   |
| **home\_directory**           | `pathlib.Path` | value of `DOMYN_SWARM_HOME`                            | Root folder for swarm state (auto-generated inside CWD).                                                                                     |                                                                   |
| **env**                 |  `dict` | `null` | A yaml dict of key values that will be set as environment variables |
| **backend**                    | `BackendConfig` | *see below*                                  | Backend specific configurations, either Slurm or Lepton                                                             |                                                                   |

### Backend Configuration: `BackendConfig`

#### SlurmConfig

| Field                         | Type           | Default                                      | Purpose                                                                                                                                      |                                                                   |
| ----------------------------- | -------------- | -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| **type** | `Literal["slurm"]` | "slurm" | Type of backend |
| **partition**                 | `str`          | `null`                           | SLURM partition to submit to.                                                                                                                |                                                                   |
| **account**                   | `str`          | `null`                               | SLURM account / charge code.                                                                                                                 |                                                                   |
| **qos** | `str` | `null` | SLURM qos where the cluster and load balancer jobs will be submitted |
| **requires_ray**              | `bool` | `null` | Set automatically to enforce the usage of Ray + vLLM for multi-node multi-gpu clusters |
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

#### SlurmEndpointConfig

| Field                  | Type  | Default      | Purpose                                                         |
| ---------------------- | ----- | ------------ | --------------------------------------------------------------- |
| **cpus\_per\_task**    | `int` | `32`          | vCPUs for the driver process (launches and monitors the swarm). |
| **mem**                | `str` | `"16GB"`     | Physical memory for the driver job.                             |
| **threads\_per\_core** | `int` | `1`          | SMT threads to request per physical core.                       |
| **wall\_time**         | `str` | `"24:00:00"` | SLURM time limit for the driver job.                            |
| **nginx\_image**       | `str \| pathlib.Path` | `null` | Path to a singularity image running NGINX as load balancer for the swarm.|
| **nginx_timeout**      | `str \| int` | "60s" | HTTP timeout for NGINX proxy requests to model replicas. |
| **port**           | `int` | `9000`       | External port exposed by the NGINX load balancer. |
| **poll_interval**  | `int` | `10` | Seconds between status checks while waiting for the load balancer to become ready. |


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
| `DOMYN_SWARM_HOME`      | path                     | Home/state directory for domyn‑swarm files (e.g., `~/.domyn_swarm`). |
| `DOMYN_SWARM_DEFAULTS`  | path (optional)          | Path to YAML with overridable defaults used by the defaults loader (e.g. `~/.domyn_swarm/defaults.yaml`).  |
| `API_TOKEN`             | secret string (optional) | API token to authenticate with the vLLM‑compatible server.           |
| `DOMYN_SWARM_MAIL_USER` | string (optional)        | Email address for Slurm job notifications.                           |
| `LEPTONAI_API_TOKEN`    | secret string (optional) | Token for Lepton API authentication.                                 |
| `LEPTON_WORKSPACE_ID`   | string (optional)        | Default Lepton workspace ID used for deployments/jobs.               |

## Python API (Programmatic usage)

> [!NOTE]
> This API is in constant evolution and you can expect breaking changes up to the final stable release

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
        output_column_name: str | list[str] = "result",
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
            output_column_name=output_column_name,
            checkpoint_interval=checkpoint_interval,
            max_concurrency=max_concurrency,
            retries=retries,
            timeout=timeout,
            **extra_kwargs,
        )
        # Our job returns 3 values per item
        self.output_column_name = ["completion", "score", "current_model"]

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
from examples.scripts.custom_job import MyCustomSwarmJob
from rich import print as rprint

config_path = Path("examples/configs/deepseek_r1_distill.yaml")
input_path = Path("examples/data/completion.parquet")
output_path = Path("results/output.parquet")
config = DomynLLMSwarmConfig.read(config_path)

# This will allocate the resources and then submit the job
with DomynLLMSwarm(cfg=config) as swarm:
    job = MyCustomSwarmJob(
            endpoint=swarm.endpoint,
            model=swarm.model,
            # 16 concurrent requests to the LLM
            max_concurrency=16,  
            # You can add custom keyword arguments, which you
            # can reference in you transform implementation by calling
            # self.kwargs
            temperature=0.2
    )
    rprint(job.to_kwargs())

    swarm.submit_job(job, input_path=input_path, output_path=output_path)
```

### Run a custom script

```shell
# Make sure the class you've implemented is on the path

# If you're using uv
PYTHONPATH=. uv run examples/scripts/custom_main.py

# or

PYTHONPATH=. python examples/scripts/custom_main.py
```

### Use in a slurm script to be submitted with sbatch

```slurm
#SBATCH various options

# You can either run the command directly
domyn-swarm job submit .....

# Or run a script which is using the programmatic API
python path/to/custom_script.py
```

You can find more examples in [examples/api](examples/api)

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

* **502 Bad Gateway Error**
  A possible reason why this happens is that the nginx http requests time out before one of the model replicas is able to return a response. In such a case, a possible fix is to increase the timeout time by including the following in your YAML config:
  ```yaml
  driver:
     nginx_timeout: "8h"
  ```

---

Happy swarming!
