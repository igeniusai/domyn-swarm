from dataclasses import dataclass
import os
import pathlib
import subprocess
import tempfile
import time
from typing import Optional, TypeVar
import jinja2
from itertools import cycle
from shutil import get_terminal_size
from threading import Thread
from time import sleep
import socket
import requests
from rich import print as rprint


DataclassT = TypeVar("DataclassT")


@dataclass
class DomynLLMSwarmConfig:
    # model / revision --------------------------------------------------------
    model: str
    revision: str | None = None

    # resources ---------------------------------------------------------------
    instances: int = 4  # number of *worker* nodes (vLLM)
    gpus_per_node: int = 4
    cpus_per_task: int = 8
    mem_per_cpu: str = "40G"
    partition: str = "boost_usr_prod"
    account: str = "iGen_train"

    # container images --------------------------------------------------------
    vllm_image: str | pathlib.Path = pathlib.Path("/leonardo_work/iGen_train/fdambro1/images/vllm_0.9.0.1.sif")
    nginx_image: str | pathlib.Path = pathlib.Path("/leonardo_work/iGen_train/fdambro1/images/nginx-dask.sif")

    # user driver -------------------------------------------------------------
    driver_script: pathlib.Path = pathlib.Path(
        "./driver.py"
    )  # must exist on a shared FS

    log_directory: pathlib.Path = pathlib.Path(
        os.path.join(os.getcwd(), "logs")
    )  # where to write slurm logs

    # misc --------------------------------------------------------------------
    max_concurrent_requests: int = 2_000
    shared_dir: pathlib.Path = pathlib.Path("/leonardo_work/iGen_train/shared")
    poll_interval: int = 10  # sacct polling cadence (s)

    # template path (auto-filled after clone)
    template_path: pathlib.Path = (
        pathlib.Path(__file__).with_suffix("").parent / "templates" / "llm_swarm.sh.j2"
    )
    nginx_template_path: pathlib.Path = (
        pathlib.Path(__file__).with_suffix("").parent / "templates" / "nginx.conf.j2"
    )
    hf_home = pathlib.Path("/leonardo_work/iGen_train/shared_hf_cache/")
    vllm_args: str = ""
    vllm_port: int = 8000
    ray_port: int = 6379
    ray_dashboard_port: int = 8265
    venv_path: pathlib.Path = pathlib.Path(os.getcwd()) / ".venv"


def is_job_running(job_id: str):
    """Given job id, check if the job is in eunning state (needed to retrieve hostname from logs)"""
    command = "squeue --me --states=R | awk '{print $1}' | tail -n +2"
    my_running_jobs = subprocess.run(
        command, shell=True, text=True, capture_output=True
    ).stdout.splitlines()
    return job_id in my_running_jobs


class Loader:
    def __init__(
        self, desc="Loading...", end="✅ Done!", failed="❌ Aborted!", timeout=0.1
    ):
        """
        A loader-like context manager
        Modified from https://stackoverflow.com/a/66558182/6611317

        Args:
            desc (str, optional): The loader's description. Defaults to "Loading...".
            end (str, optional): Final print. Defaults to "Done!".
            failed (str, optional): Final print on failure. Defaults to "Aborted!".
            timeout (float, optional): Sleep time between prints. Defaults to 0.1.
        """
        self.desc = desc
        self.end = end + " " + self.desc
        self.failed = failed + " " + self.desc
        self.timeout = timeout

        self._thread = Thread(target=self._animate, daemon=True)
        self.steps = ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]
        self.done = False

    def start(self):
        self._thread.start()
        return self

    def _animate(self):
        try:
            for c in cycle(self.steps):
                if self.done:
                    break
                rprint(f"\r{c} {self.desc}", flush=True, end="")
                sleep(self.timeout)
        except KeyboardInterrupt:
            self.stop()
            rprint("KeyboardInterrupt by user")

    def __enter__(self):
        self.start()

    def stop(self):
        self.done = True
        cols = get_terminal_size((80, 20)).columns
        rprint("\r" + " " * cols, end="", flush=True)
        rprint(f"\r{self.end}", flush=True)

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is None:
            self.stop()
        else:
            self.done = True
            cols = get_terminal_size((80, 20)).columns
            rprint("\r" + " " * cols, end="", flush=True)
            rprint(f"\r{self.failed}", flush=True)


class DomynLLMSwarm:
    """
    Context manager that:
      1. renders and submits one Slurm job
      2. waits for it to COMPLETED/FAILED via sacct
      3. cleans up on error or ^C

    Inside the allocation:
      • SLURM_NODEID 0 runs the LB (nginx) + user driver
      • SLURM_NODEID 1…instances run the vLLM servers
    """

    def __init__(self, cfg: DomynLLMSwarmConfig):
        self.cfg = cfg
        self.jobid: Optional[int] = None
        self.head_node: Optional[str] = None  # the node where the LB is running
        self._cleaned = False
        self.endpoint: Optional[str] = None  # LB endpoint, set after job submission

        # sanity check
        if not self.cfg.driver_script.is_file():
            raise FileNotFoundError(
                f"driver_script not found: {self.cfg.driver_script}"
            )
        if not self.cfg.template_path.is_file():
            raise FileNotFoundError(
                f"template_path not found: {self.cfg.template_path}"
            )

        os.makedirs(self.cfg.log_directory, exist_ok=True)

    def __enter__(self):
        self._submit_job()
        self._wait_for_running_job()
        return self  # nothing else to expose in Mode B

    def __exit__(self, exc_type, exc, tb):
        pass

    def _submit_job(self):
        ts = int(time.time())
        job_name = f"llm-swarm-{ts}"
        hosts_file = self.cfg.shared_dir / f"swarm_{ts}.hosts"

        # render the SBATCH script
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.cfg.template_path.parent),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        script_txt = env.get_template(self.cfg.template_path.name).render(
            cfg=self.cfg,
            job_name=job_name,
            hosts_file=str(hosts_file),
        )

        # write to temp file
        with tempfile.NamedTemporaryFile(
            "w", delete=False, delete_on_close=False, suffix=".sbatch"
        ) as fh:
            fh.write(script_txt)
            script_path = fh.name

        # submit
        out = subprocess.check_output(
            ["sbatch", "--parsable", script_path], text=True
        ).strip()
        # sbatch --parsable returns "<jobid>;<array_task_id>"
        self.jobid = int(out.split(";")[0])
        rprint(f"[LLMSwarm] submitted Slurm job {self.jobid}")

    def _get_head_node(self) -> str:
        nodespec = subprocess.check_output(
            ["squeue", "-j", str(self.jobid), "-h", "-O", "NodeList:2048"],
            text=True,
        ).strip()
        head_node = subprocess.check_output(
            ["scontrol", "show", "hostnames", nodespec],
            text=True,
        ).splitlines()[0]
        return head_node

    def _wait_for_running_job(self):
        if self.jobid is None:
            raise RuntimeError("No job submitted")
        rprint("[LLMSwarm] waiting for job to finish …")
        while True:
            out = subprocess.check_output(
                ["sacct", "-j", str(self.jobid), "-n", "-X", "-o", "State"], text=True
            )
            if not out:
                rprint(
                    f"[LLMSwarm] job {self.jobid} not found in sacct output, waiting …"
                )
                time.sleep(self.cfg.poll_interval)
                continue
            state = out.strip().split()[0]
            if state in {"PENDING", "RUNNING"}:
                rprint(f"[LLMSwarm] job {self.jobid} is still pending, waiting …")
                if state == "RUNNING":
                    self.head_node = self._get_head_node()
                    rprint(
                        f"[LLMSwarm] job {self.jobid} is running, waiting for vLLM to be up on the head node ({self.head_node})…"
                    )
                    try:
                        response = requests.get(
                            f"http://{self.head_node}:{self.cfg.vllm_port}/v1/models",
                            timeout=10,
                        )
                        if response.status_code == 200:
                            rprint(f"[LLMSwarm] vLLM is up on {self.head_node}")
                            self.endpoint = f"http://{self.head_node}:{self.cfg.vllm_port}"
                            return
                    except requests.ConnectionError:
                        rprint(f"[LLMSwarm] vLLM not ready yet, waiting …")
            elif state in {"COMPLETED", "FAILED", "CANCELLED"}:
                rprint(f"[LLMSwarm] job {self.jobid} finished with state: {state}")
                if state == "COMPLETED":
                    rprint("[LLMSwarm] job completed successfully")
                else:
                    rprint(f"[LLMSwarm] job failed with state: {state}")
                return
            else:
                rprint(
                    f"[LLMSwarm] job {self.jobid} finished with unknown state: {state}"
                )
                return
            time.sleep(self.cfg.poll_interval)

    def cleanup(self):
        if self._cleaned or self.jobid is None:
            return
        subprocess.run(["scancel", str(self.jobid)], check=False)
        rprint(f"[LLMSwarm] scancel {self.jobid}")
        self._cleaned = True
