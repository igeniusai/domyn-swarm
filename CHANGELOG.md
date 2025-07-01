## v0.5.2 (2025-07-01)

### Fix

- add --exclusive=user to job allocation in sbatch scripts
- fix exceptions silenced when retried, add --exclusive when submitting sbatch jobs

## v0.5.1 (2025-07-01)

### Fix

- fix ChatCompletionJob to properly handle array of messages

## v0.5.0 (2025-07-01)

### Feat

- add resource configuration for the driver/lb task
- add support for running jobs in multithreaded mode

### Fix

- fix usage of new cli argument for nthreads
- fix issue when users filter df inside transform
- fix typo
- actually fix streaming of stdout to terminal
- fix how log_directory config is after home_directory
- add cli parameters to domyn_swarm.run_job so that it's more usable
- use subprocess.PIPE for stderr for running srun commands
- fix properly the usage of a moddel saved in a local folder. If the cfg.model is a folder, then it will be mounted by the vllm containers
- quick fix to make sure that models saved in folders are actually readable by the vllm container
- update deepseek_r1.yaml

### Perf

- :zap: improve deepseek R1 performances

## v0.4.0 (2025-06-27)

### Feat

- add perplexitymixin to enable computation of perplexity by users
- :sparkles: enable tuple unpacking when when fn is called in SwarmJob.batched

### Fix

- typo introduced in previous commit
- :bug: fix a bug in jobs having multiple output columns
- :bug: remove delete_on_close=False incompatible with python 3.10
- start fixing reverse proxy, update example configs
- improve docs, fix an issue with job kwargs incorrectly handled

## v0.3.0 (2025-06-24)

### Feat

- add custom input and output columns, add deepseek r1 distill config, use vllm-only for serving models on a single node

### Fix

- add retry mechanism and update deepseek r1 config
- fix issues with checkpoints not happening, tune deepseek_r1 deployment, update example datasets

## v0.2.1 (2025-06-19)

### Fix

- **cfg**: fix default value of venv_path to None

## v0.2.0 (2025-06-18)

### Feat

- **cli**: :sparkles: add submit job and submit script commands
- **cli**: add up command to allocate the clusters without running any script
- **cfg**: add time_limit, exclude_nodes and node_list configurations
- add load balancer and proper replicas management for deployed clusters
- add replicas to config
- **deploy**: :bricks: add reverse proxy flag after cluster is deployed
- **cfg**: :zap: add new config keys
- **cli**: :sparkles: add cli interface with typer
- :rocket: initial commit

### Fix

- various fixes: remove driver_script from config, add home_directory key
- **jobs**: :bug: fix several bugs with launching jobs on already available clusters
- use proper conditionals when defining exclude_nodes and node_list
- **cfg**: rename instances to nodes in configuration and code
- **deploy**: :wrench: fix issue with nginx reverse proxy

### Refactor

- :beers: remove unneeded code
