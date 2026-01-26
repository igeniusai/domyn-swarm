import pytest
from rich.console import Console
import typer

from domyn_swarm.utils.cli import _pick_one


def test_pick_one_valid_choice(monkeypatch):
    """Selects a valid choice and returns the matching name.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    console = Console(record=True)
    monkeypatch.setattr(typer, "prompt", lambda _: "2")
    assert _pick_one(["swarm-a", "swarm-b", "swarm-c"], console) == "swarm-b"


def test_pick_one_invalid_choice_raises(monkeypatch):
    """Rejects out-of-range user input with BadParameter.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    console = Console(record=True)
    monkeypatch.setattr(typer, "prompt", lambda _: "9")
    with pytest.raises(typer.BadParameter):
        _pick_one(["swarm-a", "swarm-b"], console)
