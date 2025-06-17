import json
import pathlib
import subprocess
from typing import Optional

import yaml
from domyn_swarm import DomynLLMSwarmConfig, DomynLLMSwarm
import typer
from typing_extensions import Annotated

from domyn_swarm.helpers import launch_reverse_proxy

app = typer.Typer()


def _load_swarm_config(
    config_file: typer.FileText,
    *,
    replicas: int = 1,
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


def _start_swarm(
    name: Optional[str],
    cfg: DomynLLMSwarmConfig,
    *,
    submit_driver: bool = False,
    reverse_proxy: bool = False,
) -> None:
    """Common context‐manager + reverse proxy logic."""
    with DomynLLMSwarm(name, cfg) as swarm:
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
    ] = 1,
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


@app.command("status", short_help="Check the status of the swarm allocation")
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


@app.command("pool", short_help="Deploy a pool of swarm allocations")
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
    ids = json.loads(state_file.read_text())
    lb = ids["lb_job_id"]
    arr = ids["array_job_id"]

    typer.echo(f"🔴  Cancelling LB  job {lb}")
    subprocess.run(["scancel", str(lb)], check=False)

    typer.echo(f"🔴  Cancelling array job {arr}")
    subprocess.run(["scancel", str(arr)], check=False)

    typer.echo("✅  Swarm shutdown request sent.")


if __name__ == "__main__":
    app()
