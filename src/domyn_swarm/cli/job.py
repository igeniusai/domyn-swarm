import logging
from pathlib import Path
from typing import List, Optional

import typer

from domyn_swarm import DomynLLMSwarm, utils
from domyn_swarm.config.swarm import _load_swarm_config
from domyn_swarm.core.swarm import _load_job
from domyn_swarm.helpers.logger import setup_logger

logger = setup_logger("domyn_swarm.cli", level=logging.INFO)

job_app = typer.Typer(help="Submit a workload to a Domyn-Swarm allocation.")


@job_app.command("submit-script")
def submit_script(
    script_file: Path = typer.Argument(..., exists=True, readable=True),
    config: Optional[typer.FileText] = typer.Option(
        None,
        "-c",
        "--config",
        exists=True,
        help="YAML that defines/creates a new swarm",
    ),
    name: str | None = typer.Option(
        None, "-n", "--name", exists=True, help="Swarm name."
    ),
    args: List[str] = typer.Argument(None, help="extra CLI args passed to script"),
):
    """
    Run an *arbitrary* Python file inside the swarm head node.
    """
    if config is not None and name is not None:
        logger.error("Either --config or --jobid must be provided, not both.")
        raise typer.Exit(1)

    if config:
        cfg = _load_swarm_config(config)
        with DomynLLMSwarm(cfg=cfg) as swarm:
            swarm.submit_script(script_file, extra_args=args)

    elif name is None:
        raise RuntimeError("State is null")

    else:
        swarm: DomynLLMSwarm = DomynLLMSwarm.from_state(deployment_name=name)
        swarm.submit_script(script_file, extra_args=args)


@job_app.command("submit")
def submit_job(
    job_class: str = typer.Argument(
        default="domyn_swarm.jobs:ChatCompletionJob",
        help="Job class to run, in the form `module:ClassName`",
    ),
    input: Path = typer.Option(
        ..., "--input", exists=True, click_type=utils.ClickEnvPath()
    ),
    output: Path = typer.Option(..., "--output", click_type=utils.ClickEnvPath()),
    input_column: str = typer.Option("messages", "--input-column"),
    output_column: str = typer.Option("results", "--output-column"),
    job_kwargs: str = typer.Option(
        "{}", "--job-kwargs", help="JSON dict forwarded to job constructor"
    ),
    config: Optional[typer.FileText] = typer.Option(
        None, "-c", "--config", exists=True, help="YAML that starts a fresh swarm"
    ),
    name: Optional[str] = typer.Option(None, "-n", "--name", help="Swarm name."),
    checkpoint_dir: Path = typer.Option(
        ".checkpoints/",
        "--checkpoint-dir",
        "-cd",
        help="Directory to store checkpoints (default: .checkpoint/, no checkpoints)",
    ),
    checkpoint_interval: int = typer.Option(
        32,
        "--checkpoint-interval",
        "-ci",
        help="Batch size for processing input DataFrame (default: 32)",
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
    Submit a strongly-typed job to the swarm for DataFrame processing.

    This command processes input data through a SwarmJob (DataFrame-in → DataFrame-out)
    within the swarm environment. Jobs are defined by a class that handles the actual
    processing logic, and the input/output are managed as DataFrames with configurable
    column mappings.

    The job can either create a new swarm from a configuration file or connect to an
    existing swarm by name. Progress is tracked through checkpoints, and the operation
    supports concurrent processing with configurable retry logic.
    """
    if config is not None and name is not None:
        logger.error("Either --config or --name must be provided, not both.")
        raise typer.Exit(1)

    if config:
        cfg = _load_swarm_config(config)
        swarm_ctx = DomynLLMSwarm(cfg=cfg)
        try:
            with swarm_ctx as swarm:
                job = _load_job(
                    job_class,
                    job_kwargs,
                    endpoint=swarm.endpoint,
                    model=swarm.model,
                    checkpoint_interval=checkpoint_interval,
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
                    checkpoint_dir=checkpoint_dir,
                )
        except KeyboardInterrupt:
            abort = typer.confirm(
                "KeyboardInterrupt detected. Do you want to cancel the swarm allocation?"
            )
            if abort:
                try:
                    swarm_ctx.cleanup()
                except Exception:
                    pass
                typer.echo("Swarm allocation cancelled by user")
                raise typer.Abort()
            else:
                typer.echo("Continuing to wait for job to complete …")
    elif name is None:
        raise RuntimeError("Swarm name is null.")

    else:
        swarm = DomynLLMSwarm.from_state(deployment_name=name)
        job = _load_job(
            job_class,
            job_kwargs,
            endpoint=swarm.endpoint,
            model=swarm.model,
            checkpoint_interval=checkpoint_interval,
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
            checkpoint_dir=checkpoint_dir,
        )
