import time
from typing import Optional

from rich.console import Console
from rich.status import Status

from domyn_swarm.backends.serving.slurm_driver import SlurmDriver
from domyn_swarm.platform.http_probe import wait_http_200
from domyn_swarm.platform.protocols import ServingHandle
from domyn_swarm.platform.readiness import ServingReadiness

SLURM_BAD_STATES = {
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "BOOT_FAIL",
    "CANCELLED",
    "NODE_FAIL",
}

SLURM_WAIT_STATES = {"PENDING", "CONFIGURING"}


class SlurmReadiness(ServingReadiness):
    """
    Slurm readiness: wait for replica array & LB jobs to be RUNNING,
    resolve LB node, then poll LB HTTP /v1/models for 200.
    """

    def __init__(
        self,
        *,
        driver: SlurmDriver,
        endpoint_port: int,
        poll_interval_s: float = 10.0,
        console: Optional[Console] = None,
    ):
        self.driver = driver
        self.endpoint_port = endpoint_port
        self.poll_interval_s = poll_interval_s
        self.console = console or Console()

    def wait_ready(self, handle: ServingHandle, timeout_s: int) -> ServingHandle:
        jobid = handle.meta.get("jobid")
        lb_jobid = handle.meta.get("lb_jobid")
        if jobid is None or lb_jobid is None:
            raise RuntimeError(
                "SlurmReadiness requires 'jobid' and 'lb_jobid' in handle.meta"
            )

        with self.console.status(
            "[bold green]Waiting for LB and replicas to start..."
        ) as status:
            self._wait_jobs_running(jobid, lb_jobid, status)
            lb_node = handle.meta.get("lb_node") or self.driver.get_node_from_jobid(
                lb_jobid
            )
            handle.meta["lb_node"] = lb_node
            status.update(f"[yellow]LB job running on {lb_node}, probing HTTP…")

            base = f"http://{lb_node}:{self.endpoint_port}"
            wait_http_200(
                f"{base}/v1/models",
                timeout_s=timeout_s,
                poll_interval_s=self.poll_interval_s,
            )
            handle.url = base
            self.console.print(f"[bold green]LB healthy → {handle.url}")
            return handle

    def _wait_jobs_running(self, jobid: int, lb_jobid: int, status: Status):
        while True:
            rep = self.driver.get_job_state(jobid)
            lb = self.driver.get_job_state(lb_jobid)

            if rep == "UNKNOWN" or lb == "UNKNOWN":
                status.update(
                    f"[yellow]squeue UNKNOWN for job {jobid} or {lb_jobid}, retrying …"
                )
            elif rep in SLURM_BAD_STATES:
                raise RuntimeError(f"Replica array ended in {rep}")
            elif lb in SLURM_BAD_STATES:
                raise RuntimeError(f"LB job ended in {lb}")
            elif rep in SLURM_WAIT_STATES:
                status.update("[yellow]Waiting for replicas to start …")
            elif lb in SLURM_WAIT_STATES:
                status.update("[yellow]Waiting for LB job to start …")
            elif rep == "RUNNING" and lb == "RUNNING":
                return
            time.sleep(self.poll_interval_s)
