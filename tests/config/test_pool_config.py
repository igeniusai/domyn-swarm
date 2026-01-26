import domyn_swarm.config.pool as pool_mod
from domyn_swarm.core.swarm import DomynLLMSwarm


def test_swarm_pool_from_config(monkeypatch, tmp_path):
    """Builds a SwarmPool from a YAML configuration."""
    config_path = tmp_path / "pool.yaml"
    config_path.write_text(
        "pool:\n"
        "  - name: swarm-a\n"
        "    config_path: /tmp/cfg-a.yaml\n"
        "  - name: swarm-b\n"
        "    config_path: /tmp/cfg-b.yaml\n"
    )

    monkeypatch.setattr(pool_mod.DomynLLMSwarmConfig, "read", lambda p: f"cfg:{p}")
    monkeypatch.setattr(pool_mod, "DomynLLMSwarm", lambda cfg: object.__new__(DomynLLMSwarm))

    pool = pool_mod.SwarmPool.from_config(config_path)
    assert all(isinstance(s, DomynLLMSwarm) for s in pool.swarms)
