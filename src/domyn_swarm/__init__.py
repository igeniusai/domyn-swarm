from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import logging
from domyn_swarm import utils
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Generator, Optional
import jinja2
import requests
import typer
from rich import print as rprint
from rich.syntax import Syntax

from domyn_swarm.helpers import (
    generate_swarm_name,
    is_folder,
    path_exists,
    setup_logger,
    to_path,
)
from domyn_swarm.jobs import SwarmJob
from pydantic import BaseModel, ValidationInfo, computed_field, field_validator, Field
import shlex

from domyn_swarm.models.swarm import DomynLLMSwarmConfig

logger = setup_logger("domyn_swarm.models.swarm", level=logging.INFO)


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

    name: Optional[str] = Field(default_factory=generate_swarm_name)
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
            return cls.model_fields[info.field_name].get_default(
                call_default_factory=True
            )
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
            cfg=self.cfg,
            job_name=job_name,
            path_exists=path_exists,
            is_folder=is_folder,
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

    def submit_script(self, script_path: utils.EnvPath, detach: bool = False):
        """
        Submit a user script to the swarm allocation.
        The script will be run on the head node (SLURM_NODEID 0).
        """
        if self.jobid is None:
            raise RuntimeError("No job submitted yet")
        if not script_path.is_file():
            raise FileNotFoundError(f"Script not found: {script_path}")

        logger.info(f"Submitting user script {script_path} to job {self.jobid}")

        cmd = [
            "srun",
            "--jobid",
            str(self.lb_jobid),
            f"--nodelist={self.lb_node}",
            "--ntasks=1",
            "--overlap",
            f"--mem={self.cfg.driver.mem}",
            f"--cpus-per-task={self.cfg.driver.cpus_per_task}",
            f"--export=ALL,ENDPOINT={self.endpoint},MODEL={self.model}",
            "--overlap",
            "--ntasks=1",
        ]
        if self.cfg.mail_user:
            cmd.append(f"--mail-user={self.cfg.mail_user}")
            cmd.append("--mail-type=END,FAIL")
        
        exe = [
            f"{self.cfg.venv_path / 'bin' / 'python'}",
            str(script_path),
        ]
        cmd.extend(exe)

        if detach:
            proc = subprocess.Popen(
                cmd,
                stdout=sys.stdout,
                stderr=sys.stderr,
                start_new_session=True,
                close_fds=True,
            )
            logger.info(f"Detached process with PID {proc.pid}")
            return proc.pid
        else:
            subprocess.run(
                cmd,
                check=True,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )

    def _persist(self):
        state_file = utils.EnvPath(self.cfg.home_directory) / f"swarm_{self.jobid}.json"
        state_file.write_text(self.model_dump_json(indent=2))
        logger.info(f"State saved to {state_file}")

    def _submit_clusters_job(self):
        job_name = self.name

        self.jobid = self._submit_replicas(job_name)
        logger.info(
            f"Submitted Slurm job {self.jobid} with {self.cfg.replicas} replicas"
        )
        self.lb_jobid = self._submit_lb(job_name)
        logger.info(f"Submitted LB job for {self.lb_jobid}")
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
            try:
                out = subprocess.check_output(
                    ["squeue", "-j", str(jid), "-h", "-o", "State"],
                    text=True,
                ).strip()
                return out.split()[0] if out else "UNKNOWN"
            except Exception:
                return "UNKNOWN"

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
                        try:
                            self.lb_node = self._get_lb_node()
                            status.update(
                                f"[yellow]LB job running on {self.lb_node}, probing …"
                            )
                        except Exception:
                            continue

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
        input_path: utils.EnvPath | str,
        output_path: utils.EnvPath | str,
        num_threads: int = 1,
        detach: bool = False,
        limit: int | None = None,
        mail_user: Optional[str] = None,
    ) -> int | None:
        """
        Launch a serialized :class:`~domyn_swarm.SwarmJob` inside the current
        SLURM swarm allocation.

        The *job* object is converted to keyword arguments via
        :py:meth:`SwarmJob.to_kwargs`, transmitted to the head node
        (where ``SLURM_NODEID == 0``), reconstructed by
        ``domyn_swarm.run_job``, and executed under ``srun``.

        Parameters
        ----------
        job : SwarmJob
            The job instance to execute.
        input_path : utils.EnvPath | str
            Parquet file produced by the upstream pipeline stage.
        output_path : utils.EnvPath | str
            Destination Parquet file to be written by *job*.
        num_threads : int, default 1
            Number of CPU threads the job may use in the worker process.
        detach : bool, default False
            If *True*, start the job in a new process group and return
            immediately with its PID; if *False* (default) the call blocks
            until completion.
        limit : int or None, optional
            Maximum number of rows to read from *input_path* — handy for
            dry-runs and debugging.  When *None* (default) the entire
            dataset is processed.

        Returns
        -------
        int or None
            *detach=True*  → PID of the spawned process.
            *detach=False* → ``None`` (the call blocks).

        Raises
        ------
        RuntimeError
            The swarm manager is not ready (`self.jobid` or `self.endpoint`
            is ``None``).
        FileNotFoundError
            *input_path* does not exist.
        subprocess.CalledProcessError
            Propagated when the synchronous ``srun`` command exits with a
            non-zero status code.

        Notes
        -----
        The constructed command is logged with *rich* for transparency, e.g.::

            srun --jobid=<...> --nodelist=<...> --ntasks=1 --overlap ...
                python -m domyn_swarm.run_job --job-class=<module:Class> ...

        Examples
        --------
        >>> swarm.submit_job(
        ...     my_job,
        ...     input_path=Path("batch.parquet"),
        ...     output_path=Path("predictions.parquet"),
        ...     num_threads=4
        ... )
        """
        input_path: utils.EnvPath = to_path(input_path)
        output_path: utils.EnvPath = to_path(output_path)

        if self.jobid is None or self.endpoint is None:
            raise RuntimeError("Swarm not ready")

        if not input_path.is_file():
            raise FileNotFoundError(input_path)

        job_class = f"{job.__class__.__module__}:{job.__class__.__qualname__}"
        job_kwargs = json.dumps(job.to_kwargs())

        if self.cfg.venv_path and self.cfg.venv_path.is_dir():
            python_interpreter = self.cfg.venv_path / "bin" / "python"
        else:
            python_interpreter = sys.executable

        cmd = [
            "srun",
            f"--jobid={self.lb_jobid}",
            f"--nodelist={self.lb_node}",
            "--ntasks=1",
            "--overlap",
            "--export=ALL",  # Keep this to preserve the default env
            f"--mem={self.cfg.driver.mem}",
            f"--cpus-per-task={self.cfg.driver.cpus_per_task}",
        ]

        if mail_user or self.cfg.mail_user:
            if mail_user:
                self.cfg.mail_user = mail_user
            cmd.append(f"--mail-user={self.cfg.mail_user}")
            cmd.append("--mail-type=END,FAIL")
        
        exe = [
            str(python_interpreter),
            "-m",
            "domyn_swarm.run_job",
            f"--job-class={job_class}",
            f"--model={self.model}",
            f"--input-parquet={input_path}",
            f"--output-parquet={output_path}",
            f"--endpoint={self.endpoint}",
            f"--nthreads={num_threads}",
            "--job-kwargs",
            job_kwargs,
        ]

        if limit:
            exe.append(f"--limit={limit}")
        
        cmd.extend(exe)

        logger.info(f"Submitting job {job.__class__.__name__} to swarm {self.jobid}:")

        full_cmd = shlex.join(cmd)
        syntax = Syntax(
            full_cmd,
            "bash",
            line_numbers=False,
            word_wrap=True,
            indent_guides=True,
            padding=1,
        )
        rprint(syntax)
        if detach:
            proc = subprocess.Popen(
                cmd,
                stdout=sys.stdout,
                stderr=sys.stderr,
                start_new_session=True,
                close_fds=True,
            )
            logger.info(f"Detached process with PID {proc.pid}")
            return proc.pid
        else:
            subprocess.run(cmd, check=True, stdout=sys.stdout, stderr=sys.stderr)

    @classmethod
    def from_state(cls, state_file: utils.EnvPath) -> "DomynLLMSwarm":
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

        logger.info(f"Loaded state from {self.jobid} and {self.lb_jobid}")
        return self

    def cleanup(self):
        subprocess.run(["scancel", str(self.jobid)], check=False)
        subprocess.run(["scancel", str(self.lb_jobid)], check=False)
        logger.info(f"scancel {self.jobid} and {self.lb_jobid} requested")
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


@contextmanager
def create_swarm_pool(
    *configs_or_swarms: list[DomynLLMSwarmConfig] | list[DomynLLMSwarm],
    max_workers=None,
) -> Generator[tuple[DomynLLMSwarm], Any, None]:
    """
    You can use this utility function like this:

    ```
    with create_swarm_pool(cfg1, cfg2, cfg3, max_workers=3) as (sw1, sw2, sw3):
        sw1.submit_job()
        sw2.submit_job()
        sw3.submit_job()
    ```

    or

    ```
    with create_swarm_pool(*my_cfg_list, max_workers=5) as swarms:
        for swarm in swarms:
            swarm.submit_job()
    ```

    """
    # 1) instantiate all the context‐manager objects
    if configs_or_swarms and isinstance(configs_or_swarms[0], DomynLLMSwarmConfig):
        cms = [DomynLLMSwarm(cfg=config) for config in configs_or_swarms]
    elif configs_or_swarms and isinstance(configs_or_swarms[0], DomynLLMSwarm):
        cms = configs_or_swarms
    else:
        raise ValueError(
            "configs_or_swarms must be either a sequence of DomynLLMSwarmConfig or DomynLLMSwarm"
        )

    entered = []
    try:
        # 2) call each __enter__ in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            # submit each cm.__enter__(); collect futures keyed by cm
            futures = {exe.submit(cm.__enter__): cm for cm in cms}
            for future in as_completed(futures):
                cm = futures[future]
                res = future.result()  # will re‐raise if __enter__ failed
                entered.append((cm, res))

        # 3) yield the tuple of entered results in the original order
        #    (filter out any that didn’t make it into 'entered' if one failed)
        #    Note: if one __enter__ raised, we jump straight to finally, cleaning up
        yield tuple(res for _, res in entered)

    finally:
        # 4) tear them all down, in reverse order of successful entry
        #    pass None for exc_type, exc_val, tb if no exception
        for cm, _ in reversed(entered):
            try:
                cm.__exit__(None, None, None)
            except Exception:
                pass
