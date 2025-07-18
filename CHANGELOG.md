## v0.11.1 (2025-07-18)

### Fix

- :bug: add error log for nginx and add lest_conn to load balancer config

## v0.11.0 (2025-07-17)

### Feat

- add enable_proxy_buffering option to driver config
- add --mail-user flag to submit job command

### Refactor

- add srun_builder module
- move state manager, slurm driver and lb health checker to their own modules in core package
- :construction: start refactoring DomynSwarm core logic
- :fire: remove unneeded LLMClient and related classes, keeping AsyncOpenAIClient
- :construction: refactor helpers module into its own package
- :construction: refactory jobs structure and its package

## v0.10.0 (2025-07-11)

### Feat

- add mail_user to configuration to enable mail notifications via slurm

### Fix

- enable log stats
- fix nodes allocation of ray workers

## v0.9.1 (2025-07-08)

### Fix

- fix issue with NOT_GIVEN used as default for openai timeout
- fix bug in batch execution progress bar
- fix status check for load balancer in domyn-swarm status

## v0.9.0 (2025-07-08)

### Feat

- add status command

### Fix

- improve logging
- fix issue with default template path

## v0.8.5 (2025-07-07)

### Fix

- fix issue with exceptions retries not being logged properly by tenacity

## v0.8.4 (2025-07-07)

### Fix

- use extra_body parameter in openai client to make sure to pass all kwargs to vllm

## v0.8.3 (2025-07-07)

### Fix

- fix issue after refactoring

### Refactor

- move config models to own package

## v0.8.2 (2025-07-07)

### Refactor

- refactor cli package structure

## v0.8.1 (2025-07-04)

### Fix

- fix issue with allocation of nodes for workers in swarms with multiple replicas

## v0.8.0 (2025-07-04)

### Feat

- implement pool command to spin up multiple clusters with different configs
- add --detach flag to domyn-swarm submit job

### Fix

- fix issue increasing wait time for ray workers
- add to_path helper
- fix progress bar batch request execution in last batch

## v0.7.0 (2025-07-03)

### Feat

- add swarm pool data models

### Fix

- improve robustness for checking slurm jobs state, add create_swarm_pool utility
- add PID returned if job is submitted as detached
- rmeove unneeded files
- add --exclusive to load balancer template
- add --exclusive to load balancer template
- use output path as part of checkpointing naming convention
- use model to differentiate between checkpoints on the same source dataset
- fix issue when domyn-swarm python script is running in sbatch script

## v0.6.0 (2025-07-02)

### Feat

- add MultiTurnChatCompletion job

### Fix

- use squeue instead of sacct for checking slurm job status, add reasoning_content key to MultiTurnChatCompletionJob

## v0.5.3 (2025-07-02)

### Fix

- fix issue with srun not being able to be executed inside lb node
- fix checkpointing
- fix example data to mirror actual implementation of ChatCompletionJob

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
