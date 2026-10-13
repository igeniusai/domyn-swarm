# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""CLI, configuration, and endpoint helpers for the end-to-end scenarios."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import urllib.request
import uuid

import yaml

from .cluster import cancel, queued_jobs, wait_for

FAKE_MODEL = "e2e/fake-model"
CLI = Path(sys.executable).with_name("domyn-swarm")
UP_TIMEOUT_S = 0.8 * int(os.environ.get("DOMYN_SWARM_E2E_TIMEOUT_S", "600"))
NGINX_START_SCRIPT = (
    "# Start script of the stand-in nginx image; see shims/fake_singularity.py.\n"
    "exec nginx -g 'daemon off;' -c /etc/nginx/nginx.conf\n"
)


@dataclass(frozen=True)
class Swarm:
    name: str
    config_name: str
    endpoint: str


def tier() -> str:
    return os.environ.get("DOMYN_SWARM_E2E_TIER", "fast")


def model() -> str:
    return os.environ["E2E_VLLM_MODEL"] if tier() == "vllm" else FAKE_MODEL


def parse_swarm_name(stdout: str, config_name: str) -> str:
    names = re.findall(rf"^{re.escape(config_name)}-[0-9a-z]{{10}}$", stdout, re.MULTILINE)
    assert names, f"no swarm name for {config_name} in `up` output:\n{stdout}"
    return names[-1]


def write_images(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "python.sif").write_text("stand-in image; the singularity shim ignores it\n")
    (directory / "nginx.sif").write_text(NGINX_START_SCRIPT)
    return directory


def swarm_config(*, name: str, images: Path) -> dict:
    config: dict = {
        "name": name,
        "model": model(),
        "image": str(images / "python.sif"),
        "replicas": 2,
        "gpus_per_replica": 1,
        "gpus_per_node": 2,
        "cpus_per_task": 1,
        "args": "",
        "env": {"HF_HOME": os.environ.get("HF_HOME", "/shared/hf")},
        "wait_endpoint_s": 600,
        "watchdog": {
            "probe_interval": 2,
            "readiness_timeout": 600,
            "restart_policy": "on-failure",
            "restart_backoff_initial": 1,
            "max_restarts": 3,
        },
        "backend": {
            "type": "slurm",
            "partition": "debug",
            "account": "e2e",
            "qos": "normal",
            "time_limit": "01:00:00",
            "endpoint": {
                "nginx_image": str(images / "nginx.sif"),
                "port": 9000,
                "cpus_per_task": 1,
                "mem": "512M",
                "wall_time": "01:00:00",
            },
        },
    }
    if tier() == "vllm":
        config["args"] = (
            # -O0 skips torch.compile, which needs a C++ compiler and Python headers.
            f"--revision {os.environ['E2E_VLLM_REVISION']} --max-model-len 512 --dtype float32 -O0"
        )
        config["env"].update({"VLLM_CPU_KVCACHE_SPACE": "1", "VLLM_CPU_OMP_THREADS_BIND": "nobind"})
    return config


def cli(*args: str, timeout: float = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(CLI), *args], capture_output=True, text=True, timeout=timeout, check=False
    )


def status(name: str) -> dict:
    result = cli("status", name, "-o", "json")
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def slurm_job_id(status_payload: dict, role: str) -> str:
    jobs = {job["role"]: str(job["jobid"]) for job in status_payload["slurm_jobs"]}
    return jobs[role]


def chat(endpoint: str, model_name: str) -> dict:
    body = {
        "model": model_name,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 8,
        # vLLM's CPU random sampler is always torch.compile'd; greedy decoding is not.
        "temperature": 0,
    }
    request = urllib.request.Request(
        f"{endpoint}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def watchdog_rows(name: str) -> dict[int, dict]:
    db = Path(os.environ["DOMYN_SWARM_HOME"]) / "swarms" / name / "watchdog.db"
    with sqlite3.connect(db) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT replica_id, node, port, pid, state, http_ready "
            "FROM replica_status WHERE swarm_id = ?",
            (name,),
        ).fetchall()
    return {row["replica_id"]: dict(row) for row in rows}


class SwarmFactory:
    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir
        self.started: list[Swarm] = []
        self.config_names: list[str] = []

    def up(self, **overrides: object) -> Swarm:
        config_name = f"e2e-{uuid.uuid4().hex[:6]}"
        self.config_names.append(config_name)
        config = {**swarm_config(name=config_name, images=write_images(self.workdir)), **overrides}
        path = self.workdir / f"{config_name}.yaml"
        path.write_text(yaml.safe_dump(config))

        result = cli("up", "-c", str(path), timeout=UP_TIMEOUT_S)
        assert result.returncode == 0, f"up failed:\n{result.stdout}\n{result.stderr}"
        name = parse_swarm_name(result.stdout, config_name)
        swarm = Swarm(name=name, config_name=config_name, endpoint=status(name)["endpoint"])
        self.started.append(swarm)
        return swarm

    def tear_down(self) -> None:
        for swarm in self.started:
            # A hung `down` must not skip the scancel below.
            with contextlib.suppress(subprocess.TimeoutExpired):
                cli("down", swarm.name, "-y")
        leftover = [
            job_id
            for job_id, job_name in queued_jobs()
            if any(job_name.startswith(prefix) for prefix in self.config_names)
        ]
        cancel(*leftover)
        wait_for(lambda: not queued_jobs(), 120, "jobs still queued after teardown")
