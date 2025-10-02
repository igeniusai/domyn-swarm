import subprocess
from dataclasses import dataclass
from typing import Optional

import requests
from requests import RequestException

from domyn_swarm.backends.serving.slurm_driver import SlurmDriver
from domyn_swarm.backends.serving.slurm_readiness import (
    SLURM_BAD_STATES,
    SLURM_WAIT_STATES,
    SlurmReadiness,
)
from domyn_swarm.config.slurm import SlurmConfig
from domyn_swarm.platform.protocols import (
    ServingBackend,
    ServingHandle,
    ServingPhase,
    ServingStatus,
)


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

    def create_or_update(self, name: str, spec: dict, extras: dict) -> ServingHandle:
        replicas = spec.get("replicas", 1)
        nodes = spec.get("nodes", 1)
        gpus_per_node = spec.get("gpus_per_node", 1)
        gpus_per_replica = spec.get("gpus_per_replica", 1)
        replicas_per_node = spec.get("replicas_per_node", 1)

        jobid = self.driver.submit_replicas(
            name, replicas, nodes, gpus_per_node, gpus_per_replica, replicas_per_node
        )
        lb_jobid = self.driver.submit_endpoint(name, jobid, replicas)
        return ServingHandle(
            id=str(lb_jobid),
            url="",  # filled in wait_ready()
            meta={
                "jobid": jobid,
                "lb_jobid": lb_jobid,
                "port": self.cfg.endpoint.port,
                "name": name,
            },
        )

    def wait_ready(
        self, handle: ServingHandle, timeout_s: int, extras: dict
    ) -> ServingHandle:
        # Delegate to your health checker which sets endpoint when LB is alive
        probe = self.readiness or SlurmReadiness(
            driver=self.driver,
            endpoint_port=self.cfg.endpoint.port,
            poll_interval_s=self.cfg.endpoint.poll_interval,
        )
        return probe.wait_ready(handle, timeout_s)

    def delete(self, handle: ServingHandle) -> None:
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

    def status(self, handle: ServingHandle) -> ServingStatus:
        rep = self.driver.get_job_state(handle.meta["jobid"])
        lb = self.driver.get_job_state(handle.meta["lb_jobid"])

        # 1) Scheduler view
        if rep in SLURM_BAD_STATES or lb in SLURM_BAD_STATES:
            return ServingStatus(
                ServingPhase.FAILED, handle.url, {"rep": rep, "lb": lb}
            )
        if (
            rep in SLURM_WAIT_STATES
            or lb in SLURM_WAIT_STATES
            or rep == "UNKNOWN"
            or lb == "UNKNOWN"
        ):
            return ServingStatus(
                ServingPhase.PENDING, handle.url, {"rep": rep, "lb": lb}
            )

        # 2) Endpoint probe (non-blocking, small timeout)
        lb_node = handle.meta.get("lb_node") or self.driver.get_node_from_jobid(
            handle.meta["lb_jobid"]
        )
        base = f"http://{lb_node}:{self.cfg.endpoint.port}"
        try:
            r = requests.get(f"{base}/v1/models", timeout=1.5)
            http_ok = r.status_code == 200
            # Optional: verify expected model is listed
            model_ok = False
            try:
                data = r.json()
                names = {
                    m.get("id") for m in data.get("data", []) if isinstance(m, dict)
                }
                expected = handle.meta.get("model")  # if you stored it
                model_ok = (expected in names) if expected else http_ok
            except Exception:
                model_ok = http_ok
        except RequestException:
            http_ok = model_ok = False

        if http_ok and model_ok:
            # cache url if we didn’t earlier
            if not handle.url:
                handle.url = base
            return ServingStatus(
                ServingPhase.RUNNING,
                handle.url or base,
                {"rep": rep, "lb": lb, "http": 200},
            )

        # Slurm says RUNNING but HTTP not ready → still initializing
        return ServingStatus(
            ServingPhase.INITIALIZING,
            handle.url,
            {"rep": rep, "lb": lb, "http": "unready"},
        )
