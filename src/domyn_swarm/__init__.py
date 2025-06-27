import importlib
import json
import os
import pathlib
import random
import string
import subprocess
import sys
import tempfile
import time
from typing import Optional
import jinja2
import requests
import typer
from rich import print as rprint
import yaml

from domyn_swarm.jobs import SwarmJob
from pydantic import BaseModel, ValidationInfo, computed_field, field_validator


class DomynLLMSwarmConfig(BaseModel):
    # model / revision --------------------------------------------------------
    model: str
    revision: str | None = None

    # resources ---------------------------------------------------------------
    replicas: int = 1  # number of cluster replicas (vLLM servers)
    nodes: int = 4  # number of *worker* nodes on each replica (vLLM)
    gpus_per_node: int = 4
    cpus_per_task: int = 8
    mem_per_cpu: str = "40G"
    partition: str = "boost_usr_prod"
    account: str = "iGen_train"

    # container images --------------------------------------------------------
    vllm_image: str | pathlib.Path = pathlib.Path(
        "/leonardo_work/iGen_train/fdambro1/images/vllm_0.9.1.sif"
    )
    nginx_image: str | pathlib.Path = pathlib.Path(
        "/leonardo_work/iGen_train/fdambro1/images/nginx-dask.sif"
    )
    lb_wait: int = 1200  # seconds to wait for LB to be ready
    lb_port: int = 9000

    home_directory: pathlib.Path = pathlib.Path(
        os.path.join(os.getcwd(), ".domyn_swarm")
    )  # where to mount the home directory inside the container

    log_directory: pathlib.Path = home_directory / "logs"

    # misc --------------------------------------------------------------------
    max_concurrent_requests: int = 2_000
    poll_interval: int = 10  # sacct polling cadence (s)

    # template path (auto-filled after clone)
    template_path: pathlib.Path = (
        pathlib.Path(__file__).with_suffix("").parent / "templates" / "llm_swarm.sh.j2"
    )
    nginx_template_path: pathlib.Path = (
        pathlib.Path(__file__).with_suffix("").parent / "templates" / "nginx.conf.j2"
    )
    hf_home: pathlib.Path = pathlib.Path("/leonardo_work/iGen_train/shared_hf_cache/")
    vllm_args: str = ""
    vllm_port: int = 8000
    ray_port: int = 6379
    ray_dashboard_port: int = 8265
    venv_path: pathlib.Path | None = None

    time_limit: str = "36:00:00"  # e.g. 36 hours
    exclude_nodes: str | None = None  # e.g. "node[1-3]" (optional)
    node_list: str | None = None  # e.g. "node[4-6]" (optional)

    def model_post_init(self, context):
        os.makedirs(self.log_directory, exist_ok=True)
        os.makedirs(self.home_directory, exist_ok=True)
        return super().model_post_init(context)

    @classmethod
    def read(cls, path: pathlib.Path) -> "DomynLLMSwarmConfig":
        return _load_swarm_config(path.open())


def _load_swarm_config(
    config_file: typer.FileText, *, replicas: int | None = None
) -> DomynLLMSwarmConfig:
    """Load YAML, inject driver_script if given, apply replicas override."""
    cfg_dict = yaml.safe_load(config_file)
    cfg = DomynLLMSwarmConfig(**cfg_dict)
    # override default only if user passed something truthy
    if replicas:
        cfg.replicas = replicas
    return cfg


def is_job_running(job_id: str):
    """Given job id, check if the job is in eunning state (needed to retrieve hostname from logs)"""
    command = "squeue --me --states=R | awk '{print $1}' | tail -n +2"
    my_running_jobs = subprocess.run(
        command, shell=True, text=True, capture_output=True
    ).stdout.splitlines()
    return job_id in my_running_jobs


class DomynLLMSwarm(BaseModel):
    """
    Context manager that:
      1. renders and submits one Slurm job as a job array (each job is a replica of the same cluster)
      2. waits for it to COMPLETED/FAILED via sacct
      3. cleans up on error or ^C

    Inside the allocation:
      • SLURM_NODEID 0 runs the LB (nginx) + user driver
      • SLURM_NODEID 1…nodes run the vLLM servers
    """

    name: str | None = f"domyn-swarm-{int(time.time())}-{
        ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    }"
    cfg: DomynLLMSwarmConfig
    jobid: Optional[int] = None  # Slurm job id, set after job submission
    lb_jobid: Optional[int] = None  # LB job id, set after job submission
    lb_node: Optional[str] = None  # the node where the LB is running
    endpoint: Optional[str] = None  # LB endpoint, set after job submission
    delete_on_exit: Optional[bool] = (
        False  # Delete the resources for this cluster at the end of the job
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str, info: ValidationInfo) -> str:
        if v is None:
            return cls.model_fields[info.field_name].get_default()
        return v

    @computed_field
    @property
    def model(self) -> str:
        """
        The model name, either from the config or the job submission.
        If not set, defaults to the config's model.
        """
        return self.cfg.model

    @model.setter
    def model(self, value: str):
        """
        Setter for the model name. This allows setting the model after
        the swarm has been created, e.g., when loading from a state file.
        """
        self.cfg.model = value

    def __enter__(self):
        self._submit_clusters_job()
        self._wait_for_lb_health()
        self._persist()
        return self  # nothing else to expose in Mode B

    def __exit__(self, exc_type, exc, tb):
        if self.delete_on_exit:
            self.cleanup()

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
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sbatch") as fh:
            fh.write(script_txt)
            script_path = fh.name

        # submit
        os.makedirs(self.cfg.log_directory / job_name, exist_ok=True)
        array_spec = f"0-{self.cfg.replicas - 1}%{self.cfg.replicas}"
        out = subprocess.check_output(
            ["sbatch", "--parsable", "--array", array_spec, script_path], text=True
        ).strip()
        job_id = out.split(";")[0]
        # sbatch --parsable returns "<jobid>;<array_task_id>"

        os.makedirs(self.cfg.home_directory / "swarms" / job_id, exist_ok=True)
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
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sbatch") as fh:
            fh.write(lb_script_txt)
            # submit
        out = subprocess.check_output(
            [
                "sbatch",
                "--parsable",
                "--dependency",
                f"after:{self.jobid}",
                "--export",
                f"DEP_JOBID={self.jobid}",
                fh.name,
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
                str(self.lb_jobid),
                f"--export=ALL,ENDPOINT={self.endpoint},MODEL={self.model}",
                "--overlap",
                "--ntasks=1",
                f"{self.cfg.venv_path / 'bin' / 'python'} {str(script_path)}",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    def _persist(self):
        state_file = pathlib.Path(self.cfg.home_directory) / f"swarm_{self.jobid}.json"
        state_file.write_text(self.model_dump_json(indent=2))
        rprint(f"[LLMSwarm] state saved to {state_file}")

    def _submit_clusters_job(self):
        job_name = self.name

        self.jobid = self._submit_replicas(job_name)
        rprint(
            f"[LLMSwarm] submitted Slurm job {self.jobid} with {self.cfg.replicas} replicas"
        )
        self.lb_jobid = self._submit_lb(job_name)
        rprint(f"[LLMSwarm] submitted LB job for {self.lb_jobid}")
        self._persist()

    def _get_lb_node(self) -> str:
        nodespec = subprocess.check_output(
            ["squeue", "-j", str(self.lb_jobid), "-h", "-O", "NodeList:2048"],
            text=True,
        ).strip()
        head_node = subprocess.check_output(
            ["scontrol", "show", "hostnames", nodespec],
            text=True,
        ).splitlines()[0]
        return head_node

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

        from rich.console import Console

        console = Console()

        try:
            with console.status(
                "[bold green]Waiting for LB and replicas to start..."
            ) as status:
                while True:
                    rep_state = _sacct_state(self.jobid)
                    lb_state = _sacct_state(self.lb_jobid)

                    time.sleep(poll)  # give sacct some time to update
                    if rep_state == "UNKNOWN" or lb_state == "UNKNOWN":
                        status.update(
                            f"[yellow][LLMSwarm] sacct returned UNKNOWN for job {self.jobid} or {self.lb_jobid}, retrying …"
                        )
                        time.sleep(poll)
                        continue

                    # 1) replicas must at least be RUNNING or COMPLETED
                    if rep_state in {"FAILED", "CANCELLED", "TIMEOUT"}:
                        raise RuntimeError(f"replica array ended in {rep_state}")
                    if rep_state == "PENDING":
                        status.update("[yellow]Waiting for replicas to start …")
                        time.sleep(poll)
                        continue

                    # 2) LB must reach RUNNING
                    if lb_state in {"FAILED", "CANCELLED", "TIMEOUT"}:
                        raise RuntimeError(f"LB job ended in {lb_state} state")
                    if lb_state == "PENDING":
                        status.update("[yellow]Waiting for LB job to start …")
                        time.sleep(poll)
                        continue

                    # 3) once LB RUNNING, probe its HTTP endpoint
                    if self.lb_node is None:
                        self.lb_node = self._get_lb_node()
                        status.update(
                            f"[yellow]LB job running on {self.lb_node}, probing …"
                        )

                    try:
                        url = f"http://{self.lb_node}:{lb_port}/v1/models"
                        res = requests.get(url, timeout=5)
                        if res.status_code == 200:
                            self.endpoint = f"http://{self.lb_node}:{lb_port}"
                            console.print(
                                f"[bold green][LLMSwarm] LB healthy → {self.endpoint}"
                            )
                            return
                        status.update(
                            f"[bold green]LB responded {res.status_code}, waiting …"
                        )
                    except requests.RequestException:
                        status.update("[yellow]Waiting for LB health check…")

                    time.sleep(poll)
        except KeyboardInterrupt:
            abort = typer.confirm(
                "[LLMSwarm] KeyboardInterrupt detected. Do you want to cancel the swarm allocation?"
            )
            if abort:
                self.cleanup()
                console.print("[LLMSwarm] Swarm allocation cancelled by user")
                raise typer.Abort()
            else:
                status.update("[LLMSwarm] Continuing to wait for LB health …")
        except RuntimeError as e:
            console.print(f"[red1][LLMSwarm] Error: {e}")
            console.print("[red1][LLMSwarm] Cancelling swarm allocation")
            self.cleanup()
            raise e

    def submit_job(
        self,
        job: SwarmJob,
        *,
        input_path: pathlib.Path,
        output_path: pathlib.Path,
    ) -> None:
        """
        Launch `job` inside the swarm allocation.  The job is serialized by
        its own `.to_kwargs()` and reconstructed with run_job.py on the head
        node (SLURM_NODEID 0).
        """
        if self.jobid is None or self.endpoint is None:
            raise RuntimeError("Swarm not ready")

        if not input_path.is_file():
            raise FileNotFoundError(input_path)

        os.environ["ENDPOINT"] = self.endpoint
        os.environ["MODEL"] = self.model
        os.environ["JOB_CLASS"] = (
            f"{job.__class__.__module__}:{job.__class__.__qualname__}"
        )
        os.environ["JOB_KWARGS"] = json.dumps(job.to_kwargs())
        os.environ["INPUT_PARQUET"] = str(input_path)
        os.environ["OUTPUT_PARQUET"] = str(output_path)

        exports = ",".join(
            [
                "ALL",
                "ENDPOINT",
                "MODEL",
                "JOB_CLASS",
                "JOB_KWARGS",
                "INPUT_PARQUET",
                "OUTPUT_PARQUET",
            ]
        )

        if self.cfg.venv_path and self.cfg.venv_path.is_dir():
            python_interpreter = self.cfg.venv_path / "bin" / "python"
        else:
            python_interpreter = sys.executable

        cmd = [
            "srun",
            "--jobid",
            str(self.lb_jobid),
            "--nodelist",
            self.lb_node,
            "--ntasks=1",
            "--overlap",
            f"--export={exports}",
            python_interpreter,
            "-m",
            "domyn_swarm.run_job",
        ]

        rprint(
            f"[LLMSwarm] submitting job {job.__class__.__name__} to swarm {self.jobid}:"
        )
        rprint(f"  {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    @classmethod
    def from_state(cls, state_file: pathlib.Path) -> "DomynLLMSwarm":
        """
        Load a swarm from a saved state file (swarm_*.json).
        """
        if not state_file.is_file():
            raise FileNotFoundError(f"State file not found: {state_file}")

        with state_file.open("r") as fh:
            state: dict = json.load(fh)

        jobid = state.get("jobid")
        lb_jobid = state.get("lb_jobid")
        if jobid is None or lb_jobid is None:
            raise ValueError("State file does not contain valid job IDs")

        cfg = DomynLLMSwarmConfig(**state.get("cfg", {}))
        return DomynLLMSwarm(
            name=state.get("name", f"domyn-swarm-{int(time.time())}"),
            cfg=cfg,
            jobid=jobid,
            lb_jobid=lb_jobid,
            lb_node=state.get("lb_node"),
            endpoint=state.get("endpoint"),
            model=state.get("model"),
        )

    def _load_state(
        self,
        jobid: int,
        lb_jobid: int,
        lb_node: Optional[str] = None,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
    ) -> "DomynLLMSwarm":
        """
        Load the state of an existing swarm from a state file.
        """
        self.jobid = jobid
        self.lb_jobid = lb_jobid
        self.lb_node = lb_node
        self.endpoint = endpoint
        self.model = model or self.cfg.model

        rprint(f"[LLMSwarm] Loaded state from {self.jobid} and {self.lb_jobid}")
        return self

    def cleanup(self):
        subprocess.run(["scancel", str(self.jobid)], check=False)
        subprocess.run(["scancel", str(self.lb_jobid)], check=False)
        rprint(f"[LLMSwarm] scancel {self.jobid} and {self.lb_jobid} requested")
        self._cleaned = True


def _load_job(job_class: str, kwargs_json: str, **kwargs) -> SwarmJob:
    mod, cls = job_class.split(":", 1)
    JobCls = getattr(importlib.import_module(mod), cls)
    return JobCls(**kwargs, **json.loads(kwargs_json))


def _start_swarm(
    name: Optional[str],
    cfg: "DomynLLMSwarmConfig",
    *,
    reverse_proxy: bool = False,
) -> None:
    """Common context-manager + reverse proxy logic."""
    with DomynLLMSwarm(cfg=cfg, name=name) as swarm:
        if reverse_proxy:
            from domyn_swarm.helpers import launch_reverse_proxy

            launch_reverse_proxy(
                cfg.nginx_template_path,
                cfg.nginx_image,
                swarm.lb_node,
                swarm._get_head_node(),
                int(swarm.endpoint.split(":")[2]),
                cfg.ray_dashboard_port,
            )
