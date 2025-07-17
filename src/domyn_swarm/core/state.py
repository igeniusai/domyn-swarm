# --- SwarmStateManager class ---
import json
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from domyn_swarm import DomynLLMSwarm

from domyn_swarm import utils
from domyn_swarm.helpers.logger import setup_logger
from domyn_swarm.models.swarm import DomynLLMSwarmConfig

logger = setup_logger(__name__, level=logging.INFO)


class SwarmStateManager:
    def __init__(self, swarm: "DomynLLMSwarm"):
        self.swarm = swarm

    def save(self):
        state_file = (
            utils.EnvPath(self.swarm.cfg.home_directory)
            / f"swarm_{self.swarm.jobid}.json"
        )
        state_file.write_text(self.swarm.model_dump_json(indent=2))
        logger.info(f"State saved to {state_file}")

    @classmethod
    def load(cls, state_file: utils.EnvPath) -> "DomynLLMSwarm":
        if not state_file.is_file():
            raise FileNotFoundError(f"State file not found: {state_file}")
        with state_file.open("r") as fh:
            state = json.load(fh)

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
