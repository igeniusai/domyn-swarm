# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, suppress
from typing import cast

from ..config.swarm import DomynLLMSwarmConfig
from .swarm import DomynLLMSwarm


@contextmanager
def create_swarm_pool(
    *configs_or_swarms: "DomynLLMSwarmConfig | DomynLLMSwarm",
    config_or_swarm_list: list[DomynLLMSwarmConfig] | list[DomynLLMSwarm] | None = None,
    max_workers: int | None = None,
) -> Generator[tuple["DomynLLMSwarm", ...], None, None]:
    """Enter multiple swarms concurrently and yield them in input order.

    Args:
        *configs_or_swarms: Swarm configurations or existing swarm instances.
        config_or_swarm_list: List-form alternative to positional inputs.
        max_workers: Maximum number of concurrent context entries.

    Yields:
        Entered swarm instances in the same order as the inputs.

    Raises:
        ValueError: If inputs are missing, conflicting, or unsupported.
    """
    if configs_or_swarms and config_or_swarm_list is not None:
        raise ValueError("Pass either positional inputs or config_or_swarm_list, not both")

    input_items = (
        tuple(config_or_swarm_list) if config_or_swarm_list is not None else configs_or_swarms
    )
    if input_items and isinstance(input_items[0], DomynLLMSwarmConfig):
        cms = [DomynLLMSwarm(cfg=cast(DomynLLMSwarmConfig, item)) for item in input_items]
    elif input_items and isinstance(input_items[0], DomynLLMSwarm):
        cms = [cast(DomynLLMSwarm, item) for item in input_items]
    else:
        raise ValueError(
            "configs_or_swarms must be either a sequence of DomynLLMSwarmConfig or DomynLLMSwarm"
        )

    entered: list[tuple[DomynLLMSwarm, DomynLLMSwarm]] = []
    entered_by_index: dict[int, DomynLLMSwarm] = {}
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            futures = {exe.submit(cm.__enter__): (index, cm) for index, cm in enumerate(cms)}
            for future in as_completed(futures):
                index, cm = futures[future]
                res = future.result()
                entered.append((cm, res))
                entered_by_index[index] = res

        yield tuple(entered_by_index[index] for index in range(len(cms)))

    finally:
        for cm, _ in reversed(entered):
            with suppress(Exception):
                cm.__exit__(None, None, None)
