# --- SwarmStateManager class ---
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from domyn_swarm.backends.compute.lepton import LeptonComputeBackend
from domyn_swarm.backends.compute.slurm import SlurmComputeBackend

if TYPE_CHECKING:
    from domyn_swarm import DomynLLMSwarm

from domyn_swarm import utils
from domyn_swarm.helpers.logger import setup_logger

logger = setup_logger(__name__, level=logging.INFO)


class SwarmStateManager:
    def __init__(self, swarm: "DomynLLMSwarm"):
        self.swarm = swarm

    def save(self):
        state_file = (
            utils.EnvPath(self.swarm.cfg.home_directory) / f"{self.swarm.name}.json"
        )
        state_file.write_text(self.swarm.model_dump_json(indent=2, by_alias=True))
        logger.info(f"State saved to {state_file}")

    @classmethod
    def load(cls, state_file: Path) -> "DomynLLMSwarm":
        from domyn_swarm import DomynLLMSwarm

        if not state_file.is_file():
            raise FileNotFoundError(f"State file not found: {state_file}")
        swarm = DomynLLMSwarm.model_validate_json(state_file.read_text(), by_alias=True)

        if swarm.serving_handle is None:
            raise ValueError("Swarm does not have a serving handle.")

        platform = swarm.cfg.platform

        if (
            platform == "slurm"
            and swarm.serving_handle
            and (
                swarm.serving_handle.meta.get("jobid") is None
                or swarm.serving_handle.meta.get("lb_jobid") is None
            )
        ):
            raise ValueError("State file does not contain valid job IDs")

        if (
            platform == "slurm"
            and swarm.serving_handle.meta.get("lb_node")
            and swarm.serving_handle.meta.get("lb_jobid")
        ):
            lb_jobid = swarm.serving_handle.meta.get("lb_jobid")
            lb_node = swarm.serving_handle.meta.get("lb_node")
            if lb_jobid is None or lb_node is None:
                raise ValueError("State file does not contain valid LB job info")
            backend = SlurmComputeBackend(
                cfg=swarm.cfg, lb_jobid=lb_jobid, lb_node=lb_node
            )
        elif platform == "lepton":
            backend = LeptonComputeBackend()
        else:
            raise ValueError(f"Unsupported platform: {platform}")

        swarm._deployment.compute = backend

        return swarm
