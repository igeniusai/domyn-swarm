from rich.console import Console

from domyn_swarm.cli.tui.describe_view import render_swarm_description
from domyn_swarm.cli.tui.status import render_swarm_status
from domyn_swarm.cli.tui.theme import phase_glyph, phase_style
from domyn_swarm.platform.protocols import ServingPhase, ServingStatus


def test_render_swarm_description_renders_name():
    """Renders a description panel containing the swarm name."""
    console = Console(record=True, width=120)
    render_swarm_description(
        name="my-swarm",
        backend="slurm",
        cfg={"model": "m"},
        endpoint="http://localhost",
        console=console,
    )
    text = console.export_text()
    assert "my-swarm" in text
    assert "SLURM" in text


def test_render_swarm_status_renders_phase():
    """Renders the swarm status panel with phase information."""
    console = Console(record=True, width=120)
    status = ServingStatus(phase=ServingPhase.RUNNING, url="http://localhost", detail={"http": 200})
    render_swarm_status("swarm-1", "slurm", status, console=console)
    text = console.export_text()
    assert "RUNNING" in text
    assert "swarm-1" in text


def test_phase_glyph_and_style_defaults(monkeypatch):
    """Returns defaults for unknown phases and supports ASCII mode."""
    assert phase_style("unknown") == phase_style("UNKNOWN")
    monkeypatch.setenv("DOMYN_SWARM_ASCII", "1")
    assert phase_glyph("running") == "*"
