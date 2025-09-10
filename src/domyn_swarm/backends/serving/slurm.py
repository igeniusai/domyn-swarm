from dataclasses import dataclass
from typing import Optional

from domyn_swarm.backends.serving.slurm_readiness import SlurmReadiness
from domyn_swarm.config.slurm import SlurmConfig
from domyn_swarm.core.slurm_driver import SlurmDriver
from domyn_swarm.platform.protocols import ServingBackend, ServingHandle


@dataclass
class SlurmServingBackend(ServingBackend):  # type: ignore[misc]
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
    cfg: SlurmConfig
    readiness: Optional[SlurmReadiness] = None

    def create_or_update(self, name: str, spec: dict) -> ServingHandle:
        replicas = spec.get("replicas", 1)
        nodes = spec.get("nodes", 1)
        gpus_per_node = spec.get("gpus_per_node", 1)
        gpus_per_replica = spec.get("gpus_per_replica", 1)
        replicas_per_node = spec.get("replicas_per_node", 1)

        jobid = self.driver.submit_replicas(
            name, replicas, nodes, gpus_per_node, gpus_per_replica, replicas_per_node
        )
        lb_jobid = self.driver.submit_lb(name, jobid)
        return ServingHandle(
            id=str(lb_jobid),
            url="",  # filled in wait_ready()
            meta={
                "jobid": jobid,
                "lb_jobid": lb_jobid,
                "port": self.cfg.driver.lb_port,
                "name": name,
            },
        )

    def wait_ready(self, handle: ServingHandle, timeout_s: int) -> ServingHandle:
        # Delegate to your health checker which sets endpoint when LB is alive
        probe = self.readiness or SlurmReadiness(
            driver=self.driver,
            lb_port=self.cfg.driver.lb_port,
            poll_interval_s=self.cfg.driver.poll_interval,
        )
        return probe.wait_ready(handle, timeout_s)

    def delete(self, handle: ServingHandle) -> None:
        import subprocess

        jobid = handle.meta.get("jobid")
        lb_jobid = handle.meta.get("lb_jobid")
        if jobid:
            subprocess.run(["scancel", str(jobid)], check=False)
        if lb_jobid:
            subprocess.run(["scancel", str(lb_jobid)], check=False)

    def ensure_ready(self, handle: ServingHandle | None):
        """Ensure the current serving handle is ready, or raise if not."""
        if handle is None:
            raise RuntimeError("Serving handle is null.")
        jobid = handle.meta.get("jobid")
        lb_jobid = handle.meta.get("lb_jobid")
        lb_node = handle.meta.get("lb_node")
        endpoint = handle.url
        if not all([jobid, lb_jobid, lb_node, endpoint]):
            raise RuntimeError(
                f"Swarm not ready (jobid/lb_jobid/lb_node/endpoint): {jobid}/{lb_jobid}/{lb_node}/{endpoint}"
            )
