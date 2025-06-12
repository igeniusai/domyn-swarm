
# Domyn-Swarm

Simple CLI tool to launch vLLM clusters on Slurm.

## Installation

`pip install git+ssh://git@github.com/igeniusai/domyn-swarm.git`

or, if using `uv`:

`uv add git+ssh://git@github.com/igeniusai/domyn-swarm.git`

## How to use

1. Write your driver script which is interacting with an hosted LLM, you can see an example in examples/scripts/simple_driver.py
2. Write your configuration file according to the config spec
3. Launch `domyn-swarm`, e.g.

```bash
domyn-swarm examples/scripts/simple_driver.py --config examples/configs/simple_config.yaml
```

> **_NOTE:_** Your driver will have the ENDPOINT environment variable which you can use to fetch the endpoint to use to use the vLLM APIs

And then wait for the job to be submitted and vLLM cluster to be up.

### Configuration: `DomynLLMSwarmConfig`

All runtime options for the swarm launcher live in a single YAML file that is loaded into the `DomynLLMSwarmConfig` dataclass.  
Below is an overview of every field, its purpose, and the default that will be used if you omit it.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| **model** | `str` | **required** | HF model ID or local path. |
| **revision** | `str \| null` | `null` | Git tag/commit for the model (if using HF). |
| **instances** | `int` | `4` | Number of **worker nodes** (one *vLLM* instance per node). |
| **gpus_per_node** | `int` | `4` | GPUs allocated on each worker. |
| **cpus_per_task** | `int` | `8` | vCPUs reserved per SLURM task. |
| **mem_per_cpu** | `str` | `"40G"` | Memory per CPU core (SLURM syntax). |
| **partition** | `str` | `"boost_usr_prod"` | SLURM partition to submit to. |
| **account** | `str` | `"iGen_train"` | SLURM account / charge code. |
| **vllm_image** | `str` | `/leonardo_work/iGen_train/fdambro1/images/vllm_0.9.0.1.sif` | Singularity image that runs the vLLM workers. |
| **nginx_image** | `str` | `/leonardo_work/iGen_train/fdambro1/images/nginx-dask.sif` | Image that runs the NGINX + Dask side-services. |
| **driver_script** | `pathlib.Path` | `./driver.py` | Script invoked on the **login/driver** node. Must be on a shared filesystem. |
| **log_directory** | `pathlib.Path` | `./logs` | Directory where SLURM output/error logs are written. |
| **max_concurrent_requests** | `int` | `2000` | Upper bound enforced by vLLM REST gateway. |
| **shared_dir** | `pathlib.Path` | `/leonardo_work/iGen_train/shared` | Scratch area mounted on every node. |
| **poll_interval** | `int` | `10` | Seconds between `sacct` polling cycles while waiting for jobs. |
| **template_path** | `pathlib.Path` | *(auto-filled)* | Internal path of the Jinja2 SLURM script template; no need to touch. |
| **hf_home** | `pathlib.Path` | `/leonardo_work/iGen_train/shared_hf_cache/` | HF cache dir mounted on workers. |
| **vllm_args** | `str` | `""` | Extra CLI flags passed verbatim to `python -m vllm.entrypoints.openai.api_server …`. |
| **vllm_port** | `int` | `8000` | Port where each worker’s OpenAI-compatible API listens. |
| **venv_path** | `pathlib.Path` | `./.venv` | Virtual-env used by the *driver* process (not the containers). |

---

#### Minimal example

```yaml
# llm-swarm.yaml
model: meta-llama/Meta-Llama-3-8B-Instruct
instances: 8        # scale to 8 nodes
gpus_per_node: 2    # override default
vllm_args: "--gpu-memory-utilization 0.85"
```