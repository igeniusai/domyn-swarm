# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from threading import Event
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
    with pytest.raises(ValueError), create_swarm_pool(123):
        pass


def test_create_swarm_pool_accepts_list_argument(monkeypatch):
    configs = [MagicMock(spec=DomynLLMSwarmConfig) for _ in range(2)]
    expected = [MagicMock(), MagicMock()]
    for swarm in expected:
        swarm.__enter__.return_value = swarm
        swarm.__exit__.return_value = None

    import domyn_swarm.core.swarm_pool as swarm_pool

    monkeypatch.setattr(
        swarm_pool,
        "DomynLLMSwarm",
        lambda cfg: expected[configs.index(cfg)],
    )

    with create_swarm_pool(config_or_swarm_list=configs) as swarms:
        assert swarms == tuple(expected)


def test_create_swarm_pool_rejects_both_input_forms():
    config = MagicMock(spec=DomynLLMSwarmConfig)

    with (
        pytest.raises(ValueError, match="either positional inputs or config_or_swarm_list"),
        create_swarm_pool(config, config_or_swarm_list=[config]),
    ):
        pass


def test_create_swarm_pool_preserves_input_order(monkeypatch):
    second_entered = Event()

    class FakeSwarm:
        def __init__(self, name, *, wait_for=None, signal=None):
            self.name = name
            self.wait_for = wait_for
            self.signal = signal

        def __enter__(self):
            if self.signal:
                self.signal.set()
            if self.wait_for:
                assert self.wait_for.wait(timeout=1)
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return None

    import domyn_swarm.core.swarm_pool as swarm_pool

    monkeypatch.setattr(swarm_pool, "DomynLLMSwarm", FakeSwarm)
    first = FakeSwarm("first", wait_for=second_entered)
    second = FakeSwarm("second", signal=second_entered)

    with create_swarm_pool(first, second, max_workers=2) as swarms:
        assert [swarm.name for swarm in swarms] == ["first", "second"]
