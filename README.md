
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
   gpus_per_node: 4
   cpus_per_task: 12
   mem_per_cpu: "11G"
   replicas: 2
   home_directory: ".domyn_swarm/"
   venv_path: ".venv"
   ```

   > [!NOTE] 
   > `model` can be either an HF model or a path to a directory with a model compatible with vllm. If using an HF id, make sure that the model is saved locally in your `HF_HOME`

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

* `<module>:<ClassName>` implementing `SwarmJob`, defaults to `domyn_swarm.jobs:ChatCompletionJob`
* **--input** / **--output** — Parquet files on shared filesystem
* **--job-kwargs** — JSON for the job’s constructor
* **--config** or **--state** (one only)  -  the definition or state of the cluster where the job will be submitted
* **--batch-size** - batch size of the requests to be sent to be processed. Once a batch has finished processing, the checkpoint will be updated
* **--parallel** - Number of concurrent requests to process
* **--retries** - Number of retries for failed requests


Internally uses checkpointing, batching, and retry logic.

---

### Configuration: `DomynLLMSwarmConfig`

All runtime options for the swarm launcher live in a single YAML file that is loaded into the `DomynLLMSwarmConfig` dataclass.
Below is an overview of every field, its purpose, and the default that will be used if you omit it.

| Field                         | Type           | Default                                      | Purpose                                                                                                                                      |                                                                   |
| ----------------------------- | -------------- | -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| **model**                     | `str`          | **required**                                 | HF model ID or local path. Passed verbatim to `vllm serve`; must resolve to a local directory or an offline Hugging Face model in `hf_home`. |                                                                   |
| **hf\_home**                  | `pathlib.Path` | `/leonardo_work/iGen_train/shared_hf_cache/` | Shared Hugging Face cache mounted on all workers.                                                                                            |                                                                   |
| **revision**                  | \`str          | null\`                                       | `null`                                                                                                                                       | Git tag/commit for the model (if using HF).                       |
| **replicas**                  | `int`          | `1`                                          | How many *independent* vLLM clusters to launch (useful for A/B tests).                                                                       |                                                                   |
| **nodes**                     | `int`          | `4`                                          | Worker nodes per replica (one vLLM server per node).                                                                                         |                                                                   |
| **gpus\_per\_node**           | `int`          | `4`                                          | GPUs allocated on each worker node.                                                                                                          |                                                                   |
| **cpus\_per\_task**           | `int`          | `8`                                          | vCPUs reserved per SLURM task.                                                                                                               |                                                                   |
| **mem\_per\_cpu**             | `str`          | `"40G"`                                      | Memory per CPU core (SLURM syntax).                                                                                                          |                                                                   |
| **partition**                 | `str`          | `"boost_usr_prod"`                           | SLURM partition to submit to.                                                                                                                |                                                                   |
| **account**                   | `str`          | `"iGen_train"`                               | SLURM account / charge code.                                                                                                                 |                                                                   |
| **vllm\_image**               | \`str          | pathlib.Path\`                               | `/leonardo_work/iGen_train/fdambro1/images/vllm_0.9.1.sif`                                                                                   | Singularity image for vLLM workers.                               |
| **nginx\_image**              | \`str          | pathlib.Path\`                               | `/leonardo_work/iGen_train/fdambro1/images/nginx-dask.sif`                                                                                   | Image running NGINX + Dask side-services.                         |
| **lb\_wait**                  | `int`          | `1200`                                       | Seconds to wait for the load balancer to become healthy.                                                                                     |                                                                   |
| **lb\_port**                  | `int`          | `9000`                                       | External port exposed by the NGINX load balancer.                                                                                            |                                                                   |
| **home\_directory**           | `pathlib.Path` | `./.domyn_swarm/`                            | Root folder for swarm state (auto-generated inside CWD).                                                                                     |                                                                   |
| **log\_directory**            | \`pathlib.Path | null\`                                       | `<home_directory>/logs`                                                                                                                      | Where SLURM stdout/stderr files are written.                      |
| **max\_concurrent\_requests** | `int`          | `2000`                                       | Upper bound enforced by vLLM’s OpenAI gateway.                                                                                               |                                                                   |
| **poll\_interval**            | `int`          | `10`                                         | Seconds between `sacct` polling cycles while waiting for jobs.                                                                               |                                                                   |
| **template\_path**            | `pathlib.Path` | *(auto-filled)*                              | Internal path of the Jinja2 SLURM script template; no need to modify.                                                                        |                                                                   |
| **nginx\_template\_path**     | `pathlib.Path` | *(auto-filled)*                              | Jinja2 template for the NGINX config.                                                                                                        |                                                                   |
| **vllm\_args**                | `str`          | `""`                                         | Extra CLI flags passed verbatim to `python -m vllm.entrypoints.openai.api_server …`.                                                         |                                                                   |
| **vllm\_port**                | `int`          | `8000`                                       | Port where each worker’s OpenAI-compatible API listens.                                                                                      |                                                                   |
| **ray\_port**                 | `int`          | `6379`                                       | Port for Ray’s GCS/head node inside each replica.                                                                                            |                                                                   |
| **ray\_dashboard\_port**      | `int`          | `8265`                                       | Ray dashboard (optional).                                                                                                                    |                                                                   |
| **venv\_path**                | \`pathlib.Path | null\`                                       | `null`                                                                                                                                       | Virtual-env used by the *driver* process (not the containers).    |
| **time\_limit**               | `str`          | `"36:00:00"`                                 | Overall SLURM wall-clock limit for the allocation.                                                                                           |                                                                   |
| **exclude\_nodes**            | \`str          | null\`                                       | `null`                                                                                                                                       | Nodes to exclude, e.g. `"node[001-004]"` (pass-through to SLURM). |
| **node\_list**                | \`str          | null\`                                       | `null`                                                                                                                                       | Explicit node list, e.g. `"node[005-008]"`.                       |
| **driver**                    | `DriverConfig` | *see below*                                  | Resource overrides for the lightweight “driver” job that orchestrates the swarm.                                                             |                                                                   |

#### Nested: `DriverConfig`

| Field                  | Type  | Default      | Purpose                                                         |
| ---------------------- | ----- | ------------ | --------------------------------------------------------------- |
| **cpus\_per\_task**    | `int` | `2`          | vCPUs for the driver process (launches and monitors the swarm). |
| **mem**                | `str` | `"16GB"`     | Physical memory for the driver job.                             |
| **threads\_per\_core** | `int` | `1`          | SMT threads to request per physical core.                       |
| **wall\_time**         | `str` | `"24:00:00"` | SLURM time limit for the driver job.                            |

---

*Tip:* Any field omitted from your YAML file inherits the default above, so you can keep configuration files minimal—only override what you need.

---

## Python API (Programmatic usage)

> [!NOTE]
> This API is in constant evolution and you can expect breaking changes up to the final stable release

In the `examples` folder, you can see some examples of programmatic usage of `DomynLLMSwarm` by instantiating a custom implementation of SwarmJob and how to run it via CLI or in a custom script: `examples/scripts/custom_main.py`.

### Define a custom job

```python
# custom_job.py
# You can do whatever you want inside this function, as long as it returns a pd.DataFrame
from domyn_swarm.jobs import SwarmJob
import pandas as pd
import random

class MyCustomSwarmJob(SwarmJob):

    def __init__(self, *, endpoint = None, model = "", input_column_name = "messages", output_column_name = "result", batch_size = 16, parallel = 2, retries = 5, **extra_kwargs):
        super().__init__(endpoint=endpoint, model=model, input_column_name=input_column_name, output_column_name=output_column_name, batch_size=batch_size, parallel=parallel, retries=retries, **extra_kwargs)
        self.output_column_name = ["completion", "score", "current_model"]

    async def transform(self, df: pd.DataFrame):
        """
        You can do whatever you want inside this function, as long as it returns a pd.DataFrame
        """

        async def _call(prompt: str) -> str:
            from openai.types.completion import Completion

            # Default client is pointing to the endpoint deployed by domyn-swarm and defined in the 
            # domyn-swarm config
            resp: Completion = await self.client.completions.create(
                model=self.model, prompt=prompt, **self.kwargs
            )
            temperature = self.kwargs["temperature"] 

            return resp.choices[0].text, random.random(), self.model + f"_{temperature}"
        
        await self.batched(df["messages"].tolist(), _call)
```

### Run a custom job via CLI

```shell
PYTHONPATH=. domyn-swarm submit job examples.scripts.custom_job:MyCustomJob \
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
            parallel=16,            
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
