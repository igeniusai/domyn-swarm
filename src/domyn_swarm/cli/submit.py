import logging
from typing import List, Optional

import typer

from domyn_swarm import DomynLLMSwarm, _load_job, utils
from domyn_swarm.helpers.logger import setup_logger
from domyn_swarm.models.swarm import _load_swarm_config

logger = setup_logger("domyn_swarm.cli", level=logging.INFO)

submit_app = typer.Typer(help="Submit a workload to a Domyn-Swarm allocation.")


@submit_app.command("script")
def submit_script(
    script_file: typer.FileText = typer.Argument(..., exists=True, readable=True),
    config: Optional[typer.FileText] = typer.Option(
        None,
        "-c",
        "--config",
        exists=True,
        help="YAML that defines/creates a new swarm",
    ),
    state: Optional[utils.ClickEnvPath] = typer.Option(
        None,
        "--state",
        exists=True,
        click_type=utils.ClickEnvPath(),
        help="swarm_*.json file of an existing swarm",
    ),
    args: List[str] = typer.Argument(None, help="extra CLI args passed to script"),
):
    """
    Run an *arbitrary* Python file inside the swarm head node.
    """
    if bool(config) == bool(state):
        logger.error("Either --config or --state must be provided, not both.")
        raise typer.Exit(1)

    if config:
        cfg = _load_swarm_config(config)
        with DomynLLMSwarm(cfg=cfg) as swarm:
            swarm.submit_script(script_file, extra_args=args)
    else:
        swarm: DomynLLMSwarm = DomynLLMSwarm.from_state(state)
        swarm.submit_script(script_file, extra_args=args)


@submit_app.command("job")
def submit_job(
    job_class: str = typer.Argument(
        default="domyn_swarm.jobs:ChatCompletionJob",
        help="Job class to run, in the form `module:ClassName`",
    ),
    input: utils.ClickEnvPath = typer.Option(
        ..., "--input", exists=True, click_type=utils.ClickEnvPath()
    ),
    output: utils.ClickEnvPath = typer.Option(
        ..., "--output", click_type=utils.ClickEnvPath()
    ),
    input_column: str = typer.Option("messages", "--input-column"),
    output_column: str = typer.Option("results", "--output-column"),
    job_kwargs: str = typer.Option(
        "{}", "--job-kwargs", help="JSON dict forwarded to job constructor"
    ),
    config: Optional[typer.FileText] = typer.Option(
        None, "-c", "--config", exists=True, help="YAML that starts a fresh swarm"
    ),
    state: Optional[utils.ClickEnvPath] = typer.Option(
        None,
        "--state",
        exists=True,
        click_type=utils.ClickEnvPath(),
        help="swarm_*.json of a running swarm",
    ),
    # TODO: deprecated, remove in future versions
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        "-b",
        help="Batch size for processing input DataFrame (default: 32). Deprecated, use --checkpoint-interval instead.",
    ),
    checkpoint_interval: int = typer.Option(
        32,
        "--checkpoint-interval",
        "-ci",
        help="Batch size for processing input DataFrame (default: 32)",
    ),
    # TODO: deprecated, remove in future versions
    parallel: int = typer.Option(
        32,
        "--parallel",
        "-p",
        help="Number of concurrent requests to process (default: 32). Deprecated, use --max-concurrent-requests instead.",
    ),
    max_concurrency: int = typer.Option(
        32,
        "--max-concurrency",
        "-mc",
        help="Number of concurrent requests to process (default: 32)",
    ),
    retries: int = typer.Option(
        5,
        "--retries",
        "-r",
        help="Number of retries for failed requests (default: 5)",
    ),
    timeout: float = typer.Option(
        600,
        "--timeout",
        "-t",
        help="Timeout for each request in seconds (default: 600)",
    ),
    num_threads: int = typer.Option(
        1,
        "--num-threads",
        "-nt",
        help="How many threads should be used by the driver to run the job",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        "-l",
        help="Limit the size to be read from the input dataset. Useful when debugging and testing to reduce the size of the dataset",
    ),
    detach: bool = typer.Option(
        False, "--detach", "-d", help="Detach the job from the current terminal"
    ),
    mail_user: Optional[str] = typer.Option(
        None,
        "--mail-user",
        "-m",
        help="Email address to receive job notifications. If set, email notifications will be enabled.",
    ),
):
    """
    Run a **SwarmJob** (strongly-typed DataFrame-in → DataFrame-out) inside the swarm.
    """
    if bool(config) == bool(state):
        logger.error("Either --config or --state must be provided, not both.")
        raise typer.Exit(1)
    
    if parallel is not None:
        logger.warning(
            "The --parallel option is deprecated. Use --max-concurrency instead."
        )
        max_concurrency = parallel
    
    if batch_size is not None:
        logger.warning(
            "The --batch-size option is deprecated. Use --checkpoint-interval instead."
        )
        checkpoint_interval = batch_size

    if config:
        cfg = _load_swarm_config(config)
        with DomynLLMSwarm(cfg=cfg) as swarm:
            job = _load_job(
                job_class,
                job_kwargs,
                endpoint=swarm.endpoint,
                model=swarm.model,
                # TODO: deprecated, remove in future versions
                batch_size=checkpoint_interval,
                checkpoint_interval=checkpoint_interval,
                # TODO: deprecated, remove in future versions
                parallel=max_concurrency,
                max_concurrency=max_concurrency,
                retries=retries,
                timeout=timeout,
                input_column_name=input_column,
                output_column_name=output_column,
            )
            swarm.submit_job(
                job,
                input_path=input,
                output_path=output,
                num_threads=num_threads,
                limit=limit,
                detach=detach,
                mail_user=mail_user,
            )
    else:
        swarm: DomynLLMSwarm = DomynLLMSwarm.from_state(state)
        job = _load_job(
            job_class,
            job_kwargs,
            endpoint=swarm.endpoint,
            model=swarm.model,
            # TODO: deprecated, remove in future versions
            batch_size=checkpoint_interval,
            checkpoint_interval=checkpoint_interval,
            # TODO: deprecated, remove in future versions
            parallel=max_concurrency,
            max_concurrency=max_concurrency,
            retries=retries,
            timeout=timeout,
            input_column_name=input_column,
            output_column_name=output_column,
        )
        swarm.submit_job(
            job,
            input_path=input,
            output_path=output,
            num_threads=num_threads,
            limit=limit,
            detach=detach,
            mail_user=mail_user,
        )
