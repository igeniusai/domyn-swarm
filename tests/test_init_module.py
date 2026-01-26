import domyn_swarm


def test_init_getattr_exposes_classes():
    """Provides lazy attribute access for public classes."""
    assert domyn_swarm.DomynLLMSwarm.__name__ == "DomynLLMSwarm"
    assert domyn_swarm.DomynLLMSwarmConfig.__name__ == "DomynLLMSwarmConfig"
    assert domyn_swarm.SwarmJob.__name__ == "SwarmJob"
