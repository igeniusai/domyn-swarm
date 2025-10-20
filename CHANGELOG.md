## v0.21.4 (2025-10-20)

### Fix

- :bug: fix some bugs occuring in co-located replicas using vllm

## v0.21.3 (2025-10-17)

### Fix

- improva symlink creation by showing only relevant files
- fix issue with typo symlink path
- issue with symlink creation for singularity instance logs
- **logs**: fix various issues with ray usage with new log directory structure
- fix issue with serving spec not correctly propagated when running on slurm with requires_ray set to false
- fix issue where environment variables set in config were not propagated correctly
- **cli**: :bug: fix a bug where resources weren't cleaned up when ctrl-c-ing
- **imports**: fix issue with attributeerror raised on python 3.10

### Refactor

- **logs**: :technologist: improve usability of logs by using a sane dir tree

## v0.21.2 (2025-10-14)

### Fix

- **imports**: fix issue with circular imports on Python 3.10

## v0.21.1 (2025-10-14)

### Fix

- **cli**: fix issue with using the CLI in python 3.10

## v0.21.0 (2025-10-14)

### Feat

- **cli**: add domyn-swarm swarm list command to show available swarms in db
- **cli**: update status command for a more complete report of swarm status

### Fix

- **state**: fix issue with state queries not being loaded by CLI
- **templates**: fix issue where co-located replicas weren't deployed correctly

## v0.20.2 (2025-10-06)

## v0.20.1 (2025-10-03)

### Fix

- fix typing

## v0.20.0 (2025-10-03)

### BREAKING CHANGE

- swarms are now labelled with their deployment names
- db schema changed
- the name is now specified in the config file and then used to generate the primary key
- state is now managed with a sqlite db, state files are not needed anymore. In the CLI, state file paths are replaced by the job id

### Feat

- add usage of api_token when
- add usage of vllm_api_key environment variable propagated to vllm on singularity
- add status method for Swarm and related slurm implementation
- **state**: adapt cli to the new state management
- **state**: implement new state management
- **state**: change db schema
- unify swarm name handling
- implement init defaults command
- **docs**: add apache 2.0 badge
- **docs**: add domyn-swarm logo
- add settings to enable usage of environment variables
- add lepton api token support using lepton secrets, fix lepton job submission
- add endpoint submission to lepton
- start refactoring for job execution improvements
- **state**: add state management tests
- **state**: replace state management with a sqlite db
- **exceptions**: add custom exceptions
- **state**: add state management queries
- **ci**: add python version matrix

### Fix

- use correct fallback for LeptonConfig.lepton_workspace_id
- set home_directory attribute using the settings variable DOMYN_SWARM_HOME
- fix tests and wrong swarm name being displayed in logs
- fix issue with api_token not being used and passed to lepton jo
- **lepton**: fix lepton deployment by passing the correct objects and setting the proper image from the config
- update submit_script method
- fix issue with multiple output columns not handled correctly
- remove --platform flag from cli commands
- fix job submission by using correct venv_path attribute in slurm config and by making headers to openai client optional
- **linting**: remove unused imports
- **state**: fix conflicts
- **state**: fix handle loading
- **cli**: change swarm name help
- **cli**: change job name help
- **tests**: fix pool test
- fix type checking
- **state**: add newline
- **state**: improve docstrings
- **tests**: fix state tests and add new ones
- **state**: change slurm backend validation
- **tests**: fix broken tests
- **imports**: fix leptonai imports
- fix issue when validation not working when a required field is not present in the defaults
- fix issues related to lazy imports in tests
- **logo**: use relative paths
- **logo**: remove spaces
- **logo**: add both media
- **logo**: use raw links
- **logo**: add white logo for dark themes
- **logo**: add selector to invert colors
- **logo**: remove background
- **logo**: try to invert colors
- **logo**: add white background
- fix SlurmConfig not being actually passed to SlurmComputeBackend construction
- fix test
- fix test in TestSwarmStateManager
- fix model validator for removed backends property
- fix tests after refactoring
- fix imports and use proper propagation of secret for endpoint
- fix lepton endpoints and jobs deployments
- fix missing persist while waiting for endpoints
- **state**: add newlines to comply with hooks
- **reverse_proxy**: add support for older python versions

### Refactor

- add deprecation warnings for implemented SwarmJobs, implement new api
- **cli**: replace submit_app with job_app
- implement lazy imports for leptonai imports, which are extras
- add missing file in previous commit
- use a single backend per file, in place of a list
- fix templates and update example configs
- update package structure
- :boom: add new backends implementation for configuration, supporting multiple backends in a single config
- use Deployment class to handle the deployment of the compute jobs
- add new generic platform readiness class to abstract health checks

## v0.15.0 (2025-09-10)

### Feat

- **readme**: add badges

### Fix

- fix issue with environment variables not being expanded in config

## v0.14.0 (2025-08-29)

### Feat

- **cli**: add --checkpoint-dir to CLI

### Fix

- revert vllm_use_v1 to 0
- fix issue with checkpoint dir not being propagated correctly
- **tests**: fix tests for chat completion job

### Refactor

- fix pre-commit hooks execution

## v0.13.1 (2025-08-26)

### Refactor

- add parse_reasoning parameter to ChatCompletionJob

## v0.13.0 (2025-08-26)

### Feat

- execute hooks during the ci pipeline
- add pre-commit hooks
- add preliminary ci pipeline
- update cuda, vllm version and add ray dashboard

### Fix

- remove --contain flag to vllm instance deployment
- switch to python 3.12 in the ci pipeline
- reformat project with ruff
- Delete unused import
- simplify test to ease ci
- move ci file into the workflow dir to trigger the pipeline
- remove unneeded None guard
- fix obsolete tests
- enforce pyright type checking

## v0.12.0 (2025-08-19)

### Feat

- :zap: implement replicas sharing nodes to optimize resource usage

### Fix

- remove typing import not compatible with python 3.10
- fix edge case where multiple vllms wouldn't allocate on the same node
- fix issue with usage of min in jinja2 template
- fix how tensor-parallel-size is set when running vllm serve
- fix issue where gpus weren't allocated directly with --overlap flag

## v0.11.3 (2025-07-25)

### Fix

- add None defaults to deprecated parameters
- fix issue with parse args and improve logging
- add missing configuration key in DriverConfig model

### Refactor

- introduce checkpoint_interval and max_concurrency parameters to replace batch_size and parallel

### Perf

- add driver.nginx_timeout for load balancer config

## v0.11.2 (2025-07-18)

### Fix

- fix unescaped variable in nginx config
- add additional safeguards to timeouts in nginx config
- remove nginx_error.log
- :bug: fix issue with log folder not being created
- fix tests
- disable deletion of checkpoint files

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
