import argparse
import pathlib

import yaml
from domyn_swarm import DomynLLMSwarmConfig, DomynLLMSwarm


def main():
    parser = argparse.ArgumentParser(
        description="Launch an LLM-Swarm via Slurm."
    )
    parser.add_argument(
        "driver_script",
        type=pathlib.Path,
        help="Path to the user driver script that will be run on the swarm's master node."
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to YAML config for LLMSwarmConfig"
    )
    args = parser.parse_args()

    # load and parse YAML
    cfg_dict = yaml.safe_load(open(args.config))
    cfg_dict["driver_script"] = args.driver_script
    # initialize dataclass with defaults, then override
    cfg = DomynLLMSwarmConfig(**cfg_dict)  # defaults
    with DomynLLMSwarm(cfg):
        pass  # all work happens inside the allocation

if __name__ == "__main__":
    main()