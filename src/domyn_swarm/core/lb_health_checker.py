import subprocess
import time

import requests
import typer

from typing import TYPE_CHECKING

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
        if self.swarm.jobid is None or self.swarm.lb_jobid is None:
            raise RuntimeError("Jobs not submitted")

        lb_port = self.cfg.lb_port
        poll = self.cfg.poll_interval

        from rich.console import Console

        console = Console()

        try:
            with console.status(
                "[bold green]Waiting for LB and replicas to start..."
            ) as status:
                while True:
                    rep_state = self._sacct_state(self.swarm.jobid)
                    lb_state = self._sacct_state(self.swarm.lb_jobid)

                    time.sleep(poll)
                    if rep_state == "UNKNOWN" or lb_state == "UNKNOWN":
                        status.update(
                            f"[yellow]sacct UNKNOWN for job {self.swarm.jobid} or {self.swarm.lb_jobid}, retrying …"
                        )
                        time.sleep(poll)
                        continue

                    if rep_state in {"FAILED", "CANCELLED", "TIMEOUT"}:
                        raise RuntimeError(f"replica array ended in {rep_state}")
                    if rep_state == "PENDING":
                        status.update("[yellow]Waiting for replicas to start …")
                        time.sleep(poll)
                        continue

                    if lb_state in {"FAILED", "CANCELLED", "TIMEOUT"}:
                        raise RuntimeError(f"LB job ended in {lb_state} state")
                    if lb_state == "PENDING":
                        status.update("[yellow]Waiting for LB job to start …")
                        time.sleep(poll)
                        continue

                    if self.swarm.lb_node is None:
                        try:
                            self.swarm.lb_node = self.swarm._get_lb_node()
                            status.update(
                                f"[yellow]LB job running on {self.swarm.lb_node}, probing …"
                            )
                        except Exception:
                            continue

                    try:
                        url = f"http://{self.swarm.lb_node}:{lb_port}/v1/models"
                        res = requests.get(url, timeout=5)
                        if res.status_code == 200:
                            self.swarm.endpoint = (
                                f"http://{self.swarm.lb_node}:{lb_port}"
                            )
                            console.print(
                                f"[bold green]LB healthy → {self.swarm.endpoint}"
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
