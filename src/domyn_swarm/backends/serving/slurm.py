from dataclasses import dataclass

from domyn_swarm.core.lb_health_checker import LBHealthChecker
from domyn_swarm.core.slurm_driver import SlurmDriver
from domyn_swarm.models.swarm import DomynLLMSwarmConfig
from domyn_swarm.platform.protocols import ServingBackend, ServingHandle


@dataclass
class SlurmServing(ServingBackend):  # type: ignore[misc]
    """Adapt your existing Slurm flow into the ServingBackend protocol.

    * create_or_update -> submit replicas + LB job
    * wait_ready       -> poll with LBHealthChecker, populate .url
    * delete           -> scancel both jobs

    spec dict (suggested keys):
        {
          "name": str,
          "cfg": DomynLLMSwarmConfig,   # your config object
        }
    """

    driver: SlurmDriver
    lb_checker: LBHealthChecker
    cfg: DomynLLMSwarmConfig

    def create_or_update(self, name: str, spec: dict) -> ServingHandle:
        jobid = self.driver.submit_replicas(name)
        lb_jobid = self.driver.submit_lb(name, jobid)
        lb_node = self.driver.get_node_from_jobid(lb_jobid)
        return ServingHandle(
            id=str(lb_jobid),
            url="",  # filled in wait_ready()
            meta={
                "jobid": jobid,
                "lb_jobid": lb_jobid,
                "lb_node": lb_node,
                "port": self.cfg.lb_port,
                "name": name,
            },
        )

    def wait_ready(self, handle: ServingHandle, timeout_s: int) -> ServingHandle:
        # Delegate to your health checker which sets endpoint when LB is alive
        endpoint = self.lb_checker.wait_for_lb(timeout_s)
        if endpoint is None:
            raise RuntimeError("Failed to get endpoint from LBHealthChecker")
        handle.url = endpoint
        return handle

    def delete(self, handle: ServingHandle) -> None:
        import subprocess

        jobid = handle.meta.get("jobid")
        lb_jobid = handle.meta.get("lb_jobid")
        if jobid:
            subprocess.run(["scancel", str(jobid)], check=False)
        if lb_jobid:
            subprocess.run(["scancel", str(lb_jobid)], check=False)
