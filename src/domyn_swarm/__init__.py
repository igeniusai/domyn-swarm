from dataclasses import dataclass
import json
import os
import pathlib
import random
import string
import subprocess
import tempfile
import time
from typing import Optional, TypeVar
import jinja2
from itertools import cycle
from shutil import get_terminal_size
from threading import Thread
from time import sleep
import requests
import typer
from rich import print as rprint

DataclassT = TypeVar("DataclassT")


@dataclass
class DomynLLMSwarmConfig:
    # model / revision --------------------------------------------------------
    model: str
    revision: str | None = None

    # resources ---------------------------------------------------------------
    replicas: int = 1  # number of cluster replicas (vLLM servers)
    instances: int = 4  # number of *worker* nodes on each replica (vLLM)
    gpus_per_node: int = 4
    cpus_per_task: int = 8
    mem_per_cpu: str = "40G"
    partition: str = "boost_usr_prod"
    account: str = "iGen_train"

    # container images --------------------------------------------------------
    vllm_image: str | pathlib.Path = pathlib.Path(
        "/leonardo_work/iGen_train/fdambro1/images/vllm_0.9.0.1.sif"
    )
    nginx_image: str | pathlib.Path = pathlib.Path(
        "/leonardo_work/iGen_train/fdambro1/images/nginx-dask.sif"
    )
    lb_wait: int = 1200  # seconds to wait for LB to be ready
    lb_port: int = 9000

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
      1. renders and submits one Slurm job as a job array (each job is a replica of the same cluster)
      2. waits for it to COMPLETED/FAILED via sacct
      3. cleans up on error or ^C

    Inside the allocation:
      • SLURM_NODEID 0 runs the LB (nginx) + user driver
      • SLURM_NODEID 1…instances run the vLLM servers
    """

    def __init__(self, name: str, cfg: DomynLLMSwarmConfig):
        rand_string = "".join(
            random.choices(string.ascii_uppercase + string.digits, k=6)
        )
        self.name: str = name or f"domyn-swarm-{rand_string}"
        self.cfg = cfg
        self.jobid: Optional[int] = None
        self.lb_jobid: Optional[int] = None  # LB job id, set after job submission
        self.lb_node: Optional[str] = None  # the node where the LB is running
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
        self._wait_for_lb_health()
        return self  # nothing else to expose in Mode B

    def __exit__(self, exc_type, exc, tb):
        pass

    def _submit_replicas(self, job_name: str) -> int:
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.cfg.template_path.parent),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        script_txt = env.get_template(self.cfg.template_path.name).render(
            cfg=self.cfg, job_name=job_name
        )

        # write to temp file
        with tempfile.NamedTemporaryFile(
            "w", delete=False, delete_on_close=False, suffix=".sbatch"
        ) as fh:
            fh.write(script_txt)
            script_path = fh.name

        # submit
        array_spec = f"0-{self.cfg.replicas - 1}%{self.cfg.replicas}"
        out = subprocess.check_output(
            ["sbatch", "--parsable", "--array", array_spec, script_path], text=True
        ).strip()
        job_id = out.split(";")[0]
        # sbatch --parsable returns "<jobid>;<array_task_id>"

        os.makedirs(self.cfg.log_directory / "swarms" / job_id, exist_ok=True)
        return int(job_id)

    def _submit_lb(self, job_name: str) -> int:
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.cfg.template_path.parent),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        lb_script_txt = env.get_template("lb.sh.j2").render(
            cfg=self.cfg, job_name=job_name, dep_jobid=self.jobid
        )

        # write to temp file
        with tempfile.NamedTemporaryFile(
            "w", delete=False, delete_on_close=False, suffix=".sbatch"
        ) as fh:
            fh.write(lb_script_txt)
            lb_script_path = fh.name

        # submit
        out = subprocess.check_output(
            [
                "sbatch",
                "--parsable",
                "--dependency",
                f"after:{self.jobid}",
                "--export",
                f"DEP_JOBID={self.jobid}",
                lb_script_path,
            ],
            text=True,
        ).strip()
        return int(out)

    def submit_script(self, script_path: pathlib.Path):
        """
        Submit a user script to the swarm allocation.
        The script will be run on the head node (SLURM_NODEID 0).
        """
        if self.jobid is None:
            raise RuntimeError("No job submitted yet")
        if not script_path.is_file():
            raise FileNotFoundError(f"Script not found: {script_path}")

        rprint(f"[LLMSwarm] submitting user script {script_path} to job {self.jobid}")
        # TODO: THis should be a separate sbatch job, not an srun
        subprocess.run(
            [
                "srun",
                "--jobid",
                str(self.jobid),
                f"--export=ALL,ENDPOINT={self.endpoint}",
                "--overlap",
                "--ntasks=1",
                f"{self.cfg.venv_path / 'bin' / 'python'} {str(script_path)}",
            ],
            check=True,
        )

    def _persist(self):
        state = {
            "array_job_id": self.jobid,
            "lb_job_id": self.lb_jobid,
        }
        state_file = pathlib.Path(self.cfg.log_directory) / f"swarm_{self.jobid}.json"
        state_file.write_text(json.dumps(state, indent=2))
        rprint(f"[LLMSwarm] state saved to {state_file}")

    def _submit_job(self):
        ts = int(time.time())
        job_name = f"{self.name}-{ts}"

        self.jobid = self._submit_replicas(job_name)
        rprint(
            f"[LLMSwarm] submitted Slurm job {self.jobid} with {self.cfg.replicas} replicas"
        )
        self.lb_jobid = self._submit_lb(job_name)
        rprint(f"[LLMSwarm] submitted LB job for {self.lb_jobid}")
        self._persist()

    def _get_head_node(self) -> str:
        nodespec = subprocess.check_output(
            ["squeue", "-j", str(self.lb_jobid), "-h", "-O", "NodeList:2048"],
            text=True,
        ).strip()
        head_node = subprocess.check_output(
            ["scontrol", "show", "hostnames", nodespec],
            text=True,
        ).splitlines()[0]
        return head_node

    def _wait_for_lb_health(self):
        """
        Block until *both* the replica-array **and** the Nginx load-balancer
        are up and the LB answers HTTP on /v1/models.  Sets
            • self.lb_node    – host running the LB container
            • self.endpoint   – LB URL (http://host:<lb_port>)
        """
        if self.jobid is None or self.lb_jobid is None:
            raise RuntimeError("Jobs not submitted")

        lb_port = self.cfg.lb_port
        poll = self.cfg.poll_interval

        def _sacct_state(jid: int) -> str:
            out = subprocess.check_output(
                ["sacct", "-j", str(jid), "-n", "-X", "-o", "State"],
                text=True,
            ).strip()
            return out.split()[0] if out else "UNKNOWN"

        rprint(
            f"[LLMSwarm] waiting for replicas ({self.jobid}) and LB ({self.lb_jobid}) …"
        )

        time.sleep(poll)
        while True:
            rep_state = _sacct_state(self.jobid)
            lb_state = _sacct_state(self.lb_jobid)

            try:
                if rep_state == "UNKNOWN" or lb_state == "UNKNOWN":
                    rprint(
                        f"[LLMSwarm] sacct returned UNKNOWN for job {self.jobid} or {self.lb_jobid}, retrying …"
                    )
                    time.sleep(poll)
                    continue

                # 1) replicas must at least be RUNNING or COMPLETED
                if rep_state in {"FAILED", "CANCELLED", "TIMEOUT"}:
                    raise RuntimeError(f"replica array ended in {rep_state}")
                if rep_state == "PENDING":
                    rprint("  • replicas still pending …")
                    time.sleep(poll)
                    continue

                # 2) LB must reach RUNNING
                if lb_state in {"FAILED", "CANCELLED", "TIMEOUT"}:
                    raise RuntimeError(f"LB job ended in {lb_state} state")
                if lb_state == "PENDING":
                    rprint("  • LB job still pending …")
                    time.sleep(poll)
                    continue

                # 3) once LB RUNNING, probe its HTTP endpoint
                if self.lb_node is None:
                    self.lb_node = self._get_head_node()
                    rprint(f"  • LB job running on {self.lb_node}, probing …")

                try:
                    url = f"http://{self.lb_node}:{lb_port}/v1/models"
                    res = requests.get(url, timeout=5)
                    if res.status_code == 200:
                        self.endpoint = f"http://{self.lb_node}:{lb_port}"
                        rprint(f"[LLMSwarm] LB healthy → {self.endpoint}")
                        return
                    rprint(f"  • LB responded {res.status_code}, waiting …")
                except requests.RequestException:
                    rprint("  • LB not reachable yet …")

                time.sleep(poll)
            except KeyboardInterrupt:
                abort = typer.confirm(
                    "[LLMSwarm] KeyboardInterrupt detected. Do you want to cancel the swarm allocation?"
                )
                if abort:
                    self.cleanup()
                    rprint("[LLMSwarm] Swarm allocation cancelled by user")
                    raise typer.Abort()
                else:
                    rprint("[LLMSwarm] Continuing to wait for LB health …")
            except RuntimeError as e:
                rprint(f"[LLMSwarm] Error: {e}")
                rprint("[LLMSwarm] Cancelling swarm allocation")
                self.cleanup()
                raise e

    def cleanup(self):
        if self._cleaned or self.jobid is None:
            return
        subprocess.run(["scancel", str(self.jobid)], check=False)
        subprocess.run(["scancel", str(self.lb_jobid)], check=False)
        rprint(f"[LLMSwarm] scancel {self.jobid} and {self.lb_jobid} requested")
        self._cleaned = True
