# Copyright 2025 iGenius S.p.A
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import json
import logging
from pathlib import Path

import typer

from domyn_swarm.config.swarm import _load_swarm_config
from domyn_swarm.core.swarm import DomynLLMSwarm, _load_job
from domyn_swarm.helpers.logger import setup_logger
import domyn_swarm.utils as utils

logger = setup_logger("domyn_swarm.cli", level=logging.INFO)

job_app = typer.Typer(help="Submit a workload to a Domyn-Swarm allocation.")


def _parse_json_object(value: str, *, param_name: str) -> dict:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"{param_name} must be a valid JSON object.") from exc
    if not isinstance(parsed, dict):
        raise typer.BadParameter(f"{param_name} must be a JSON object.")
    return parsed


def _maybe_parse_json_object(value: str | None, *, param_name: str) -> dict | None:
    if value is None:
        return None
    return _parse_json_object(value, param_name=param_name)


def _build_job_kwargs_json(
    *,
    job_kwargs: str,
    data_backend: str | None,
    native_backend: bool | None,
    native_batch_size: int | None,
    backend_read_kwargs: str | None,
    backend_write_kwargs: str | None,
) -> str:
    job_kwargs_dict = _parse_json_object(job_kwargs, param_name="job_kwargs")

    if data_backend:
        job_kwargs_dict["data_backend"] = data_backend
    if native_backend is not None:
        job_kwargs_dict["native_backend"] = native_backend
    if native_batch_size is not None:
        job_kwargs_dict["native_batch_size"] = native_batch_size

    read_kwargs = _maybe_parse_json_object(backend_read_kwargs, param_name="backend_read_kwargs")
    if read_kwargs is not None:
        job_kwargs_dict["backend_read_kwargs"] = read_kwargs

    write_kwargs = _maybe_parse_json_object(backend_write_kwargs, param_name="backend_write_kwargs")
    if write_kwargs is not None:
        job_kwargs_dict["backend_write_kwargs"] = write_kwargs

    return json.dumps(job_kwargs_dict)


def _submit_loaded_job(
    *,
    swarm: DomynLLMSwarm,
    job_class: str,
    job_kwargs: str,
    job_name: str | None,
    input_path: Path,
    output_path: Path,
    input_column: str,
    output_column: str,
    checkpoint_dir: Path | None,
    no_resume: bool,
    no_checkpointing: bool,
    runner: str,
    checkpoint_interval: int,
    max_concurrency: int,
    retries: int,
    timeout: float,
    num_threads: int,
    limit: int | None,
    detach: bool,
    mail_user: str | None,
) -> None:
    resolved_checkpoint_dir = (
        swarm.swarm_dir / "checkpoints" if checkpoint_dir is None else checkpoint_dir
    )
    job = _load_job(
        job_class,
        job_kwargs,
        name=job_name,
        endpoint=swarm.endpoint,
        model=swarm.model,
        checkpoint_interval=checkpoint_interval,
        max_concurrency=max_concurrency,
        retries=retries,
        timeout=timeout,
        input_column_name=input_column,
        output_cols=output_column,
    )
    swarm.submit_job(
        job,
        input_path=input_path,
        output_path=output_path,
        num_threads=num_threads,
        limit=limit,
        detach=detach,
        mail_user=mail_user,
        checkpoint_dir=resolved_checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        no_resume=no_resume,
        no_checkpointing=no_checkpointing,
        runner=runner,
    )


def _maybe_cancel_swarm_on_keyboard_interrupt(swarm_ctx: DomynLLMSwarm) -> None:
    abort = typer.confirm("KeyboardInterrupt detected. Do you want to cancel the swarm allocation?")
    if abort:
        with contextlib.suppress(Exception):
            swarm_ctx.cleanup()
        typer.echo("Swarm allocation cancelled by user")
        raise typer.Abort() from None
    typer.echo("Continuing to wait for job to complete …")


@job_app.command("submit-script")
def submit_script(
    script_file: Path = typer.Argument(..., exists=True, readable=True),
    config: typer.FileText | None = typer.Option(
        None,
        "-c",
        "--config",
        exists=True,
        help="YAML that defines/creates a new swarm",
    ),
    name: str | None = typer.Option(None, "-n", "--name", exists=True, help="Swarm name."),
    args: list[str] = typer.Argument(None, help="extra CLI args passed to script"),
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
    input: Path = typer.Option(..., "--input", exists=True, click_type=utils.ClickEnvPath()),
    output: Path = typer.Option(..., "--output", click_type=utils.ClickEnvPath()),
    input_column: str = typer.Option("messages", "--input-column"),
    output_column: str = typer.Option("results", "--output-column"),
    job_kwargs: str = typer.Option(
        "{}", "--job-kwargs", help="JSON dict forwarded to job constructor"
    ),
    data_backend: str | None = typer.Option(
        None,
        "--data-backend",
        help="Data backend for IO (pandas, polars, ray).",
    ),
    native_backend: bool | None = typer.Option(
        None,
        "--native-backend/--no-native-backend",
        help="Use native backend batches when supported.",
    ),
    native_batch_size: int | None = typer.Option(
        None,
        "--native-batch-size",
        help="Batch size for native backend mode (optional).",
    ),
    backend_read_kwargs: str | None = typer.Option(
        None,
        "--backend-read-kwargs",
        help="JSON dict forwarded to backend read() call.",
    ),
    backend_write_kwargs: str | None = typer.Option(
        None,
        "--backend-write-kwargs",
        help="JSON dict forwarded to backend write() call.",
    ),
    job_name: str | None = typer.Option(None, "--job-name", help="Optional job name for logging"),
    config: typer.FileText | None = typer.Option(
        None, "-c", "--config", exists=True, help="YAML that starts a fresh swarm"
    ),
    name: str | None = typer.Option(None, "-n", "--name", help="Swarm name."),
    checkpoint_dir: Path | None = typer.Option(
        None,
        "--checkpoint-dir",
        "-cd",
        help="Directory to store checkpoints (default: .checkpoint/, no checkpoints)",
    ),
    no_resume: bool = typer.Option(
        False,
        "--no-resume",
        "--ignore-checkpoints",
        help="Ignore existing checkpoints for this run (forces recompute).",
    ),
    no_checkpointing: bool = typer.Option(
        False,
        "--no-checkpointing",
        help="Disable checkpointing entirely (no read/write checkpoint state).",
    ),
    runner: str = typer.Option(
        "pandas",
        "--runner",
        help="Runner implementation for non-ray backends (pandas, arrow).",
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
        help="Limit the size to be read from the input dataset. "
        "Useful when debugging and testing to reduce the size of the dataset",
    ),
    detach: bool = typer.Option(
        False, "--detach", "-d", help="Detach the job from the current terminal"
    ),
    mail_user: str | None = typer.Option(
        None,
        "--mail-user",
        "-m",
        help="Email address to receive job notifications. "
        "If set, email notifications will be enabled.",
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

    job_kwargs = _build_job_kwargs_json(
        job_kwargs=job_kwargs,
        data_backend=data_backend,
        native_backend=native_backend,
        native_batch_size=native_batch_size,
        backend_read_kwargs=backend_read_kwargs,
        backend_write_kwargs=backend_write_kwargs,
    )

    if config:
        cfg = _load_swarm_config(config)
        swarm_ctx = DomynLLMSwarm(cfg=cfg)
        try:
            with swarm_ctx as swarm:
                _submit_loaded_job(
                    swarm=swarm,
                    job_class=job_class,
                    job_kwargs=job_kwargs,
                    job_name=job_name,
                    input_path=input,
                    output_path=output,
                    input_column=input_column,
                    output_column=output_column,
                    checkpoint_dir=checkpoint_dir,
                    no_resume=no_resume,
                    no_checkpointing=no_checkpointing,
                    runner=runner,
                    checkpoint_interval=checkpoint_interval,
                    max_concurrency=max_concurrency,
                    retries=retries,
                    timeout=timeout,
                    num_threads=num_threads,
                    limit=limit,
                    detach=detach,
                    mail_user=mail_user,
                )
        except KeyboardInterrupt:
            _maybe_cancel_swarm_on_keyboard_interrupt(swarm_ctx)
    elif name is None:
        raise RuntimeError("Swarm name is null.")

    else:
        swarm = DomynLLMSwarm.from_state(deployment_name=name)
        _submit_loaded_job(
            swarm=swarm,
            job_class=job_class,
            job_kwargs=job_kwargs,
            job_name=job_name,
            input_path=input,
            output_path=output,
            input_column=input_column,
            output_column=output_column,
            checkpoint_dir=checkpoint_dir,
            no_resume=no_resume,
            no_checkpointing=no_checkpointing,
            runner=runner,
            checkpoint_interval=checkpoint_interval,
            max_concurrency=max_concurrency,
            retries=retries,
            timeout=timeout,
            num_threads=num_threads,
            limit=limit,
            detach=detach,
            mail_user=mail_user,
        )
