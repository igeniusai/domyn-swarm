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

import logging
from pathlib import Path
from typing import Literal

import typer

from domyn_swarm.cli import job_helpers as helpers
from domyn_swarm.config.swarm import _load_swarm_config
from domyn_swarm.core.state.state_manager import SwarmStateManager
from domyn_swarm.core.swarm import DomynLLMSwarm
from domyn_swarm.helpers.logger import setup_logger
import domyn_swarm.utils as utils

logger = setup_logger("domyn_swarm.cli", level=logging.INFO)

job_app = typer.Typer(help="Submit a workload to a Domyn-Swarm allocation.")


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
            handle = swarm.submit_script(script_file, extra_args=args)
            helpers.emit_submission_json(
                handle=handle,
                command="submit-script",
                swarm_name=swarm.name,
            )

    elif name is None:
        raise RuntimeError("State is null")

    else:
        swarm: DomynLLMSwarm = DomynLLMSwarm.from_state(deployment_name=name)
        handle = swarm.submit_script(script_file, extra_args=args)
        helpers.emit_submission_json(
            handle=handle,
            command="submit-script",
            swarm_name=swarm.name,
        )


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
    id_column: str | None = typer.Option(
        None,
        "--id-column",
        "--id-col",
        help="Optional column name used for stable row ids.",
    ),
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
    shard_output: bool = typer.Option(
        False,
        "--shard-output",
        help="When output is a directory and using the Polars runner, write one parquet file per "
        "shard (based on --num-threads) using checkpoint outputs as the source of truth.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        "-l",
        help="Limit the size to be read from the input dataset. "
        "Useful when debugging and testing to reduce the size of the dataset",
    ),
    shard_mode: Literal["id", "index"] = typer.Option(
        "id",
        "--shard-mode",
        help="How to split input when --num-threads > 1. "
        "'id' uses stable id hashing (resume-friendly), "
        "'index' uses legacy row order sharding. "
        "Keep --num-threads fixed across resumes to avoid reshuffling shards.",
    ),
    global_resume: bool = typer.Option(
        False,
        "--global-resume",
        help="When resuming with sharded execution, filter inputs using global "
        "done ids across all shards (useful if --limit changed).",
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
    checkpoint_tag: str | None = typer.Option(
        None,
        "--checkpoint-tag",
        help="An optional tag to be used when checkpointing is enabled. "
        "It will be used in place of the default uuid-based tag.",
    ),
    ray_address: str | None = typer.Option(
        None,
        "--ray-address",
        help="Ray cluster address to connect to when --data-backend=ray (optional).",
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

    job_kwargs = helpers.build_job_kwargs_json(
        job_kwargs=job_kwargs,
        data_backend=data_backend,
        native_backend=native_backend,
        native_batch_size=native_batch_size,
        id_column_name=id_column,
        backend_read_kwargs=backend_read_kwargs,
        backend_write_kwargs=backend_write_kwargs,
    )
    run_spec = helpers.JobRunSpec(
        input_path=input,
        output_path=output,
        shard_output=shard_output,
        checkpoint_dir=checkpoint_dir,
        no_resume=no_resume,
        no_checkpointing=no_checkpointing,
        runner=runner,
        num_threads=num_threads,
        limit=limit,
        shard_mode=shard_mode,
        global_resume=global_resume,
        detach=detach,
        mail_user=mail_user,
        ray_address=ray_address,
        checkpoint_tag=checkpoint_tag,
    )

    if config:
        cfg = _load_swarm_config(config)
        swarm_ctx = DomynLLMSwarm(cfg=cfg)
        try:
            with swarm_ctx as swarm:
                job = helpers.build_job_for_swarm(
                    swarm=swarm,
                    job_class=job_class,
                    job_kwargs=job_kwargs,
                    job_name=job_name,
                    input_column=input_column,
                    output_column=output_column,
                    checkpoint_interval=checkpoint_interval,
                    max_concurrency=max_concurrency,
                    retries=retries,
                    timeout=timeout,
                )
                handle = helpers.submit_loaded_job(
                    swarm=swarm,
                    request=helpers.JobSubmitRequest(job=job, run=run_spec),
                )
                helpers.emit_submission_json(
                    handle=handle,
                    command="submit",
                    swarm_name=swarm.name,
                )
        except KeyboardInterrupt:
            helpers.maybe_cancel_swarm_on_keyboard_interrupt(swarm_ctx)
    elif name is None:
        raise RuntimeError("Swarm name is null.")

    else:
        swarm = DomynLLMSwarm.from_state(deployment_name=name)
        job = helpers.build_job_for_swarm(
            swarm=swarm,
            job_class=job_class,
            job_kwargs=job_kwargs,
            job_name=job_name,
            input_column=input_column,
            output_column=output_column,
            checkpoint_interval=checkpoint_interval,
            max_concurrency=max_concurrency,
            retries=retries,
            timeout=timeout,
        )
        handle = helpers.submit_loaded_job(
            swarm=swarm,
            request=helpers.JobSubmitRequest(job=job, run=run_spec),
        )
        helpers.emit_submission_json(
            handle=handle,
            command="submit",
            swarm_name=swarm.name,
        )


@job_app.command("wait")
def wait_job(
    job_id: str | None = typer.Option(
        None, "--job-id", help="Internal Domyn job ID from the local state DB."
    ),
    external_id: str | None = typer.Option(
        None, "--external-id", help="Provider external ID (for example Slurm step id)."
    ),
    handle_json: str | None = typer.Option(
        None,
        "--handle-json",
        help="JSON handle payload (use '-' to read from stdin). If omitted, stdin is auto-read.",
    ),
    name: str | None = typer.Option(
        None,
        "-n",
        "--name",
        help="Optional deployment name used to disambiguate --external-id or handle payload.",
    ),
    stream_logs: bool = typer.Option(
        True,
        "--stream-logs/--no-stream-logs",
        help="Stream backend logs while waiting (when supported by backend).",
    ),
) -> None:
    """Wait for a submitted job to reach a terminal state.

    Args:
        job_id: Internal job ID selector.
        external_id: Provider external-id selector.
        handle_json: Handle JSON selector, literal payload or stdin marker.
        name: Optional deployment name hint.
        stream_logs: Whether to stream backend logs while waiting.
    """
    target = helpers.resolve_job_target(
        job_id=job_id,
        external_id=external_id,
        handle_json=handle_json,
        deployment_name=name,
    )
    swarm = DomynLLMSwarm.from_state(deployment_name=target.swarm_name)
    target.handle.status = swarm.wait_job(target.handle, stream_logs=stream_logs)
    if target.job_id:
        SwarmStateManager.update_job(
            target.job_id,
            status=target.handle.status,
            external_id=target.handle.external_id,
        )
    helpers.emit_job_control_json(command="wait", target=target, status=target.handle.status)


@job_app.command("cancel")
def cancel_job(
    job_id: str | None = typer.Option(
        None, "--job-id", help="Internal Domyn job ID from the local state DB."
    ),
    external_id: str | None = typer.Option(
        None, "--external-id", help="Provider external ID (for example Slurm step id)."
    ),
    handle_json: str | None = typer.Option(
        None,
        "--handle-json",
        help="JSON handle payload (use '-' to read from stdin). If omitted, stdin is auto-read.",
    ),
    name: str | None = typer.Option(
        None,
        "-n",
        "--name",
        help="Optional deployment name used to disambiguate --external-id or handle payload.",
    ),
) -> None:
    """Cancel a submitted job.

    Args:
        job_id: Internal job ID selector.
        external_id: Provider external-id selector.
        handle_json: Handle JSON selector, literal payload or stdin marker.
        name: Optional deployment name hint.
    """
    target = helpers.resolve_job_target(
        job_id=job_id,
        external_id=external_id,
        handle_json=handle_json,
        deployment_name=name,
    )
    swarm = DomynLLMSwarm.from_state(deployment_name=target.swarm_name)
    target.handle.status = swarm.cancel_job(target.handle)

    if target.job_id:
        SwarmStateManager.update_job(
            target.job_id,
            status=target.handle.status,
            external_id=target.handle.external_id,
        )
    helpers.emit_job_control_json(command="cancel", target=target, status=target.handle.status)
