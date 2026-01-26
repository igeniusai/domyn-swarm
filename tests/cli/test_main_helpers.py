import types

import pytest
import typer

import domyn_swarm.cli.main as main_mod
from domyn_swarm.platform.protocols import ServingPhase, ServingStatus


class DummySwarm:
    """Swarm stub that tracks shutdown calls."""

    def __init__(self):
        self.down_called = False

    def status(self):
        return ServingStatus(phase=ServingPhase.RUNNING, url=None, detail=None)

    def down(self):
        self.down_called = True


def test_down_by_name_confirms(monkeypatch):
    """Prompts and shuts down a running swarm when confirmed."""
    swarm = DummySwarm()
    monkeypatch.setattr(main_mod.SwarmStateManager, "load", lambda deployment_name: swarm)
    monkeypatch.setattr(typer, "confirm", lambda *_args, **_kwargs: True)
    main_mod.down_by_name("swarm-1", yes=False)
    assert swarm.down_called is True


def test_down_by_name_aborts(monkeypatch):
    """Aborts when the user declines shutdown."""
    swarm = DummySwarm()
    monkeypatch.setattr(main_mod.SwarmStateManager, "load", lambda deployment_name: swarm)
    monkeypatch.setattr(typer, "confirm", lambda *_args, **_kwargs: False)
    with pytest.raises(typer.Exit):
        main_mod.down_by_name("swarm-1", yes=False)
    assert swarm.down_called is False


def test_down_by_config_select(monkeypatch, tmp_path):
    """Selects a swarm when multiple matches exist."""
    swarm = DummySwarm()
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("name: base\nmodel: m\nbackend: {type: slurm}\n")

    monkeypatch.setattr(
        main_mod, "_load_swarm_config", lambda file_obj: types.SimpleNamespace(name="base")
    )
    monkeypatch.setattr(
        main_mod.SwarmStateManager, "list_by_base_name", lambda base: ["swarm-a", "swarm-b"]
    )
    monkeypatch.setattr(main_mod.SwarmStateManager, "load", lambda deployment_name: swarm)
    monkeypatch.setattr(main_mod, "_pick_one", lambda names, console: names[0])
    monkeypatch.setattr(typer, "confirm", lambda *_args, **_kwargs: True)

    with pytest.raises(typer.Exit):
        main_mod.down_by_config(cfg_path.open(), yes=False, all_=False, select=True)
    assert swarm.down_called is True
