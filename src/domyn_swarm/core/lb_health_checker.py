import subprocess
import time
from typing import TYPE_CHECKING

import requests
import typer
from rich.console import Console
from rich.status import Status

if TYPE_CHECKING:
    from domyn_swarm import DomynLLMSwarm


class LBHealthChecker:
    def __init__(self, swarm: "DomynLLMSwarm"):
        self.swarm = swarm
        self.cfg = swarm.cfg

    def _sacct_state(self, jid: int) -> str:
        try:
            out = subprocess.check_output(
                ["squeue", "-j", str(jid), "-h", "-o", "State"],
                text=True,
            ).strip()
            return out.split()[0] if out else "UNKNOWN"
        except Exception:
            return "UNKNOWN"

    def wait_for_lb(self):
        """
        Wait for the load balancer and replicas to start.
        This method will block until the LB is healthy or a timeout occurs."""
        console = Console()
        with console.status(
            "[bold green]Waiting for LB and replicas to start..."
        ) as status:
            try:
                self._wait_for_jobs_to_start(status)
                self._resolve_lb_node(status)
                self._wait_for_http_ready(status, console)
            except KeyboardInterrupt:
                abort = typer.confirm(
                    "[LLMSwarm] KeyboardInterrupt detected. Do you want to cancel the swarm allocation?"
                )
                if abort:
                    self.swarm.cleanup()
                    console.print("[LLMSwarm] Swarm allocation cancelled by user")
                    raise typer.Abort()
                else:
                    status.update("[LLMSwarm] Continuing to wait for LB health …")

            except RuntimeError as e:
                console.print(f"[red1][LLMSwarm] Error: {e}")
                console.print("[red1][LLMSwarm] Cancelling swarm allocation")
                self.swarm.cleanup()
                raise e

    def _wait_for_jobs_to_start(self, status: Status):
        """
        Wait for the replica array and LB jobs to start.
        This method will block until both jobs are in a RUNNING state or an error occurs.
        """
        if self.swarm.jobid is None:
            raise RuntimeError("Job ID is null.")

        if self.swarm.lb_jobid is None:
            raise RuntimeError("LB Job ID is null.")

        lb_state = rep_state = None
        while True:
            rep_state = self._sacct_state(self.swarm.jobid)
            lb_state = self._sacct_state(self.swarm.lb_jobid)

            if rep_state == "UNKNOWN" or lb_state == "UNKNOWN":
                status.update(
                    f"[yellow]sacct UNKNOWN for job {self.swarm.jobid} or {self.swarm.lb_jobid}, retrying …"
                )
            elif rep_state in {"FAILED", "CANCELLED", "TIMEOUT"}:
                raise RuntimeError(f"replica array ended in {rep_state}")
            elif rep_state == "PENDING":
                status.update("[yellow]Waiting for replicas to start …")
            elif lb_state in {"FAILED", "CANCELLED", "TIMEOUT"}:
                raise RuntimeError(f"LB job ended in {lb_state} state")
            elif lb_state == "PENDING":
                status.update("[yellow]Waiting for LB job to start …")
            else:
                return
            time.sleep(self.cfg.poll_interval)

    def _resolve_lb_node(self, status: Status):
        """
        Resolve the load balancer node by checking the job state.
        This method will block until the LB node is resolved or an error occurs.
        """
        while self.swarm.lb_node is None:
            try:
                self.swarm.lb_node = self.swarm._get_lb_node()
                status.update(
                    f"[yellow]LB job running on {self.swarm.lb_node}, probing …"
                )
            except Exception:
                time.sleep(self.cfg.poll_interval)

    def _wait_for_http_ready(self, status: Status, console: Console):
        """
        Wait for the LB HTTP endpoint to be ready.
        This method will block until the LB responds with a 200 OK status or an error occurs
        """
        lb_port = self.cfg.lb_port
        url = f"http://{self.swarm.lb_node}:{lb_port}/v1/models"
        while True:
            try:
                res = requests.get(url, timeout=5)
                if res.status_code == 200:
                    self.swarm.endpoint = f"http://{self.swarm.lb_node}:{lb_port}"
                    console.print(f"[bold green]LB healthy → {self.swarm.endpoint}")
                    return
                status.update(f"[bold green]LB responded {res.status_code}, waiting …")
            except requests.RequestException:
                status.update("[yellow]Waiting for LB health check…")
            time.sleep(self.cfg.poll_interval)
