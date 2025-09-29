import logging
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, TypeVar, cast

import typer
import yaml

from domyn_swarm.config.settings import get_settings
from domyn_swarm.helpers.logger import setup_logger

logger = setup_logger("domyn_swarm.cli", level=logging.INFO)

init_app = typer.Typer(help="Initialize a new Domyn-Swarm configuration.")

settings = get_settings()


# ----------------------- tiny helpers -----------------------


def _load_yaml(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            return yaml.safe_load(path.read_text()) or {}
        except Exception:
            return {}
    return {}


def _save_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


T = TypeVar("T")


def _get(d: Mapping[str, Any], dotted: str, default: T | None = None) -> T | None:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    # We trust the caller’s expected type (guided by `default`)
    return cast(T | None, cur)


def _set(d: Dict[str, Any], dotted: str, value: Any):
    parts = dotted.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def _prompt_str(label: str, default: Optional[str] = None, allow_empty=False) -> str:
    while True:
        val = typer.prompt(label, default=default if default is not None else "")
        if allow_empty:
            return val
        if val.strip():
            return val.strip()
        typer.secho("Value cannot be empty.", fg=typer.colors.RED)


def _prompt_int(
    label: str,
    default: Optional[int] = None,
    min_v: Optional[int] = None,
    max_v: Optional[int] = None,
) -> int:
    while True:
        val = typer.prompt(label, default=str(default) if default is not None else "")
        try:
            n = int(val)
        except ValueError:
            typer.secho("Please enter a valid integer.", fg=typer.colors.RED)
            continue
        if (min_v is not None and n < min_v) or (max_v is not None and n > max_v):
            typer.secho(
                f"Value must be between {min_v} and {max_v}.", fg=typer.colors.RED
            )
            continue
        return n


_MEM_RE = re.compile(r"^\s*\d+\s*(GB|MB|TB)\s*$", re.IGNORECASE)


def _prompt_mem(label: str, default: str | None = "16GB") -> str:
    while True:
        v = typer.prompt(label, default=default).strip()
        if _MEM_RE.match(v):
            return v.upper()
        typer.secho("Use a size like 16GB / 32000MB / 1TB", fg=typer.colors.RED)


_TIME_RE = re.compile(r"^\d{1,3}:\d{2}:\d{2}$")  # HH:MM:SS (HH up to 999)


def _prompt_walltime(label: str, default: str | None = "24:00:00") -> str:
    while True:
        v = typer.prompt(label, default=default).strip()
        if _TIME_RE.match(v):
            return v
        typer.secho("Use HH:MM:SS (e.g., 24:00:00).", fg=typer.colors.RED)


def _yesno(label: str, default: bool = True) -> bool:
    return typer.confirm(label, default=default)


# ----------------------- interactive flow -----------------------


def _configure_slurm_defaults(existing: Dict[str, Any]) -> Dict[str, Any]:
    base = "slurm"
    base_endpoint = f"{base}.endpoint"
    typer.secho(
        "\nConfigure defaults for SLURM endpoint", fg=typer.colors.CYAN, bold=True
    )

    image = _prompt_str(
        "Path to Singularity/Container image for inference server",
        default=_get(existing, f"{base}.image", ""),
    )
    partition = _prompt_str(
        "SLURM partition", default=_get(existing, f"{base}.partition", "")
    )
    account = _prompt_str(
        "SLURM account", default=_get(existing, f"{base}.account", "")
    )
    qos = _prompt_str(
        "SLURM QoS (optional)",
        default=_get(existing, f"{base}.qos", ""),
        allow_empty=True,
    )
    nginx_image = _prompt_str(
        "Path to Nginx image (Singularity/Container)",
        default=_get(existing, f"{base}.nginx_image", ""),
    )
    port = _prompt_int(
        "LB port",
        default=_get(existing, f"{base_endpoint}.port", 9000),
        min_v=1,
        max_v=65535,
    )
    poll = _prompt_int(
        "Poll interval (seconds)",
        default=_get(existing, f"{base_endpoint}.poll_interval", 10),
        min_v=1,
        max_v=3600,
    )
    cpus = _prompt_int(
        "CPUs per task",
        default=_get(existing, f"{base_endpoint}.cpus_per_task", 2),
        min_v=1,
    )
    mem = _prompt_mem(
        "Memory per task", default=_get(existing, f"{base_endpoint}.mem", "16GB")
    )
    tpc = _prompt_int(
        "Threads per core",
        default=_get(existing, f"{base_endpoint}.threads_per_core", 1),
        min_v=1,
    )
    wall = _prompt_walltime(
        "Wall time (HH:MM:SS)",
        default=_get(existing, f"{base_endpoint}.wall_time", "24:00:00"),
    )
    proxy_buf = _yesno(
        "Enable Nginx proxy buffering?",
        default=bool(_get(existing, f"{base_endpoint}.enable_proxy_buffering", True)),
    )
    nginx_timeout = _prompt_str(
        "Nginx timeout (e.g., 60s)",
        default=str(_get(existing, f"{base_endpoint}.nginx_timeout", "60s")),
    )

    out: Dict[str, Any] = {}
    _set(out, f"{base}.image", image)
    _set(out, f"{base}.partition", partition)
    if qos.strip():
        _set(out, f"{base}.qos", qos)
    _set(out, f"{base}.account", account)
    _set(out, f"{base_endpoint}.nginx_image", nginx_image)
    _set(out, f"{base_endpoint}.port", port)
    _set(out, f"{base_endpoint}.poll_interval", poll)
    _set(out, f"{base_endpoint}.cpus_per_task", cpus)
    _set(out, f"{base_endpoint}.mem", mem)
    _set(out, f"{base_endpoint}.threads_per_core", tpc)
    _set(out, f"{base_endpoint}.wall_time", wall)
    _set(out, f"{base_endpoint}.enable_proxy_buffering", proxy_buf)
    _set(out, f"{base_endpoint}.nginx_timeout", nginx_timeout)
    return out


def _configure_lepton_defaults(existing: Dict[str, Any]) -> Dict[str, Any]:
    typer.secho(
        "\nConfigure defaults for LEPTON (DGX Cloud)", fg=typer.colors.CYAN, bold=True
    )

    out: Dict[str, Any] = {}
    # You can keep this minimal—defaults file is for *defaults*, not per-deployment specifics
    # Add only values you want to use as fallbacks in your config model.
    base = "lepton"
    base_ep = f"{base}.endpoint"
    base_job = f"{base}.job"

    # Endpoint defaults
    workspace = _prompt_str(
        "Default Lepton workspace", default=_get(existing, f"{base_ep}.workspace", "")
    )
    resource_shape = _prompt_str(
        "Default endpoint resource_shape (e.g., gpu.4xh200)",
        default=_get(existing, f"{base_ep}.resource_shape", ""),
    )
    node_group = _prompt_str(
        "Default endpoint node_group (optional)",
        default=_get(existing, f"{base_ep}.node_group", ""),
        allow_empty=True,
    )
    _set(out, f"{base}.workspace_id", workspace)
    _set(out, f"{base_ep}.resource_shape", resource_shape)
    if node_group.strip():
        _set(out, f"{base_ep}.node_group", node_group)

    # Job defaults
    job_image = _prompt_str(
        "Default job image (e.g., igeniusai/domyn-swarm:latest)",
        default=_get(existing, f"{base_job}.image", "igeniusai/domyn-swarm:latest"),
    )
    _set(out, f"{base_job}.image", job_image)

    return out


# ----------------------- CLI command -----------------------


@init_app.command(
    "defaults", help="Create a defaults.yaml configuration file to be used later."
)
def create_defaults(
    output: str = typer.Option(
        settings.home / "defaults.yaml",
        "-o",
        "--output",
        help="Path to save the defaults YAML configuration file.",
    ),
    force: bool = typer.Option(
        False, "--force", help="Overwrite existing file if it exists."
    ),
):
    """
    Interactively create a defaults.yaml used by Domyn-Swarm to prefill configuration values.
    """
    path = Path(output)
    if path.exists() and not force:
        if not _yesno(f"File '{path}' exists. Overwrite?", default=False):
            raise typer.Abort()

    existing = _load_yaml(path)

    typer.secho(
        "Which platforms do you want defaults for?", fg=typer.colors.CYAN, bold=True
    )
    want_slurm = _yesno("Configure Slurm defaults?", default=True)
    want_lepton = _yesno("Configure Lepton (DGX Cloud) defaults?", default=False)

    result: Dict[str, Any] = {}

    if want_slurm:
        result.update(_configure_slurm_defaults(existing))

    if want_lepton:
        result.update(_configure_lepton_defaults(existing))

    if not result:
        typer.secho("No sections selected; nothing to write.", fg=typer.colors.YELLOW)
        raise typer.Abort()

    # Show preview
    typer.secho("\nPreview of defaults.yaml:\n", fg=typer.colors.MAGENTA, bold=True)
    typer.echo(yaml.safe_dump(result, sort_keys=False))

    if not _yesno("Write these defaults to file?", default=True):
        raise typer.Abort()

    _save_yaml(path, result)
    typer.secho(f"Defaults written to {path}", fg=typer.colors.GREEN, bold=True)
