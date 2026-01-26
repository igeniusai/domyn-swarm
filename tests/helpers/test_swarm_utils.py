import re

import domyn_swarm.helpers.swarm as swarm_mod


def test_generate_swarm_name_respects_backend_limit(monkeypatch):
    """Generates a slug+suffix within backend length constraints.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """

    class FixedUlid:
        """Deterministic ULID stub."""

        def __str__(self) -> str:  # pragma: no cover - deterministic helper
            return "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    monkeypatch.setattr(swarm_mod, "ULID", FixedUlid)
    name = swarm_mod.generate_swarm_name("My Test Swarm!!!", "slurm")
    assert len(name) <= 63
    assert name.startswith("my-test-swarm-")
    assert re.match(r"^[a-z0-9-]+$", name)


def test_generate_swarm_name_fallback_backend(monkeypatch):
    """Uses safe default limit for unknown backends.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """

    class FixedUlid:
        """Deterministic ULID stub."""

        def __str__(self) -> str:  # pragma: no cover - deterministic helper
            return "0123456789" * 3

    monkeypatch.setattr(swarm_mod, "ULID", FixedUlid)
    name = swarm_mod.generate_swarm_name("!!", "unknown")
    assert name.startswith("swarm-")
