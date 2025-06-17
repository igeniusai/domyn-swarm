import importlib
import json
import pathlib
import subprocess
from typing import List, Optional

import yaml
from domyn_swarm import DomynLLMSwarmConfig, DomynLLMSwarm
import typer
from typing_extensions import Annotated

from domyn_swarm.helpers import launch_reverse_proxy
from domyn_swarm.job import SwarmJob
app = typer.Typer()
submit_app   = typer.Typer(help="Submit a workload to a Domyn-Swarm allocation.")
app.add_typer(submit_app, name="submit", help="Submit a workload to a Domyn-Swarm allocation.")


def _load_swarm_config(
    config_file: typer.FileText,
    *,
    replicas: int | None = None,
    driver_script: Optional[pathlib.Path] = None,
) -> DomynLLMSwarmConfig:
    """Load YAML, inject driver_script if given, apply replicas override."""
    cfg_dict = yaml.safe_load(config_file)
    if driver_script is not None:
        cfg_dict["driver_script"] = driver_script
    cfg = DomynLLMSwarmConfig(**cfg_dict)
    # override default only if user passed something truthy
    if replicas:
        cfg.replicas = replicas
    return cfg


def _load_job(job_class: str, kwargs_json: str) -> SwarmJob:
    mod, cls = job_class.split(":", 1)
    JobCls   = getattr(importlib.import_module(mod), cls)
    return JobCls(**json.loads(kwargs_json))



def _start_swarm(
    name: Optional[str],
    cfg: DomynLLMSwarmConfig,
    *,
    submit_driver: bool = False,
    reverse_proxy: bool = False,
) -> None:
    """Common context-manager + reverse proxy logic."""
    with DomynLLMSwarm(cfg=cfg, name=name) as swarm:
        if submit_driver:
            swarm.submit_script(cfg.driver_script)
        if reverse_proxy:
            launch_reverse_proxy(
                cfg.nginx_template_path,
                cfg.nginx_image,
                swarm.lb_node,
                cfg.vllm_port,
                cfg.ray_dashboard_port,
            )


@app.command("up", short_help="Launch a swarm allocation with a configuration")
def launch_up(
    config: Annotated[
        typer.FileText,
        typer.Option(
            ..., "-c", "--config", help="Path to YAML config for LLMSwarmConfig"
        ),
    ],
    reverse_proxy: Annotated[
        bool,
        typer.Option(
            "--reverse-proxy/--no-reverse-proxy",
            help="Enable reverse proxy for the swarm allocation",
        ),
    ] = False,
    name: Annotated[
        Optional[str],
        typer.Option(
            "--name",
            "-n",
            help="Name of the swarm allocation. If not provided, a random name will be generated.",
        ),
    ] = None,
    replicas: Annotated[
        Optional[int],
        typer.Option(
            "--replicas",
            "-r",
            help="Number of replicas for the swarm allocation. Defaults to 1.",
        ),
    ] = None,
):
    cfg = _load_swarm_config(config, replicas=replicas)
    _start_swarm(name, cfg, reverse_proxy=reverse_proxy)


@app.command(
    "run", short_help="Launch a swarm allocation with a driver script and configuration"
)
def launch_run(
    driver_script: Annotated[
        pathlib.Path,
        typer.Argument(
            file_okay=True,
            help="Path to the driver script to execute inside the swarm allocation",
        ),
    ],
    config: Annotated[
        typer.FileText,
        typer.Option(
            ..., "-c", "--config", help="Path to YAML config for LLMSwarmConfig"
        ),
    ],
    reverse_proxy: Annotated[
        bool,
        typer.Option(
            "--reverse-proxy/--no-reverse-proxy",
            help="Enable reverse proxy for the swarm allocation",
        ),
    ] = False,
    name: Annotated[
        Optional[str],
        typer.Option(
            "--name",
            "-n",
            help="Name of the swarm allocation. If not provided, a random name will be generated.",
        ),
    ] = None,
    replicas: Annotated[
        Optional[int],
        typer.Option(
            "--replicas",
            "-r",
            help="Number of replicas for the swarm allocation. Defaults to 1.",
        ),
    ] = 1,
):
    cfg = _load_swarm_config(
        config,
        replicas=replicas,
        driver_script=driver_script,
    )
    _start_swarm(
        name,
        cfg,
        submit_driver=True,
        reverse_proxy=reverse_proxy,
    )


@app.command("status", short_help="Check the status of the swarm allocation (not yet implemented)")
def check_status(
    name: Annotated[
        Optional[str],
        typer.Option(
            "--name",
            "-n",
            help="Name of the swarm allocation to check status for. If not provided, checks all allocations.",
        ),
    ],
):
    pass


@app.command("pool", short_help="Deploy a pool of swarm allocations from a YAML config (not yet implemented)")
def deploy_pool(
    config: Annotated[
        typer.FileText,
        typer.Argument(
            file_okay=True,
            help="Path to YAML config for a pool of swarm allocations",
        ),
    ],
):
    pass


@app.command("down", short_help="Shut down a swarm allocation")
def down(
    state_file: pathlib.Path = typer.Argument(
        ..., exists=True, help="The swarm_*.json file printed at launch"
    ),
):
    swarm = DomynLLMSwarm.model_validate_json(state_file.read_text())  # validate the file
    lb = swarm.lb_jobid
    arr = swarm.jobid

    typer.echo(f"🔴  Cancelling LB  job {lb}")
    subprocess.run(["scancel", str(lb)], check=False)

    typer.echo(f"🔴  Cancelling array job {arr}")
    subprocess.run(["scancel", str(arr)], check=False)

    typer.echo("✅  Swarm shutdown request sent.")


@submit_app.command("script")
def submit_script(
    script_file: pathlib.Path = typer.Argument(..., exists=True, readable=True),
    config:      Optional[pathlib.Path] = typer.Option(None, "-c", "--config", exists=True,
                        help="YAML that defines/creates a new swarm"),
    state:       Optional[pathlib.Path] = typer.Option(None, "--state", exists=True,
                        help="swarm_*.json file of an existing swarm"),
    args:        List[str] = typer.Argument(None, help="extra CLI args passed to script"),
):
    """
    Run an *arbitrary* Python file inside the swarm head node.
    """
    if bool(config) == bool(state):
        typer.echo("Either --config or --state must be provided, not both.", err=True)
        raise typer.Exit(1)

    if config:
        cfg = _load_swarm_config(config)
        with DomynLLMSwarm(cfg) as swarm:
            swarm.submit_script(script_file, extra_args=args)
    else:
        swarm: DomynLLMSwarm = DomynLLMSwarm.from_state(state)
        swarm.submit_script(script_file, extra_args=args)

@submit_app.command("job")
def submit_job(
    job_class:   str,
    input:       pathlib.Path = typer.Option(..., "--input", exists=True),
    output:      pathlib.Path = typer.Option(..., "--output"),
    job_kwargs:  str          = typer.Option("{}", "--job-kwargs",
                         help='JSON dict forwarded to job constructor'),
    config:      Optional[pathlib.Path] = typer.Option(None, "-c", "--config", exists=True,
                         help="YAML that starts a fresh swarm"),
    state:       Optional[pathlib.Path] = typer.Option(None, "--state", exists=True,
                         help="swarm_*.json of a running swarm"),
):
    """
    Run a **SwarmJob** (strongly-typed DataFrame-in → DataFrame-out) inside the swarm.
    """
    if bool(config) == bool(state):
        typer.echo("Either --config or --state must be provided, not both.", err=True)
        raise typer.Exit(1)

    job = _load_job(job_class, job_kwargs)

    if config:
        cfg = _load_swarm_config(config)
        with DomynLLMSwarm(cfg) as swarm:
            swarm.submit_job(job, input_path=input, output_path=output)
    else:
        swarm: DomynLLMSwarm = DomynLLMSwarm.from_state(state)
        swarm.submit_job(job, input_path=input, output_path=output)



if __name__ == "__main__":
    app()
