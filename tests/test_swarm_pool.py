from unittest.mock import MagicMock

import pytest

from domyn_swarm.config.swarm import DomynLLMSwarmConfig
from domyn_swarm.core.swarm_pool import create_swarm_pool


def test_create_swarm_pool_with_configs():
    mock_swarm = MagicMock()
    mock_swarm.__enter__.return_value = mock_swarm
    mock_swarm.__exit__.return_value = None

    # Patch DomynLLMSwarm only during instantiation
    import domyn_swarm.core.swarm_pool as swarm_pool

    original_cls = swarm_pool.DomynLLMSwarm
    swarm_pool.DomynLLMSwarm = lambda cfg: mock_swarm

    try:
        cfgs = [MagicMock(spec=DomynLLMSwarmConfig) for _ in range(3)]
        with create_swarm_pool(*cfgs, max_workers=3) as swarms:
            assert len(swarms) == 3
            for s in swarms:
                assert s is mock_swarm
    finally:
        swarm_pool.DomynLLMSwarm = original_cls


def test_create_swarm_pool_invalid_input():
    with pytest.raises(ValueError):
        with create_swarm_pool(123):
            pass
