# Copyright 2025 iGenius S.p.A
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

from rich.console import Console

import domyn_swarm.cli.tui as TUI


def _st():
    # Minimal stand-in for ServingStatus; function under test doesn't inspect internals
    return SimpleNamespace(phase="RUNNING", url="http://x", detail={"http": 200})


def test_render_status_with_single_tuple_calls_swarm_status(mocker):
    mock_swarm = mocker.patch.object(TUI, "render_swarm_status")
    mock_multi = mocker.patch.object(TUI, "render_multi_status")

    status = _st()
    console = Console(width=80)

    TUI.render_status(("my-swarm", "slurm", status), console=console)

    mock_swarm.assert_called_once_with(
        "my-swarm", "slurm", status, replica_summary=None, console=console
    )
    mock_multi.assert_not_called()


def test_render_status_with_single_item_iterable_calls_swarm_status(mocker):
    mock_swarm = mocker.patch.object(TUI, "render_swarm_status")
    mock_multi = mocker.patch.object(TUI, "render_multi_status")

    status = _st()
    console = Console()

    TUI.render_status([("solo", "lepton", status)], console=console)

    mock_swarm.assert_called_once_with(
        "solo", "lepton", status, replica_summary=None, console=console
    )
    mock_multi.assert_not_called()


def test_render_status_with_multiple_items_calls_multi_status(mocker):
    mock_swarm = mocker.patch.object(TUI, "render_swarm_status")
    mock_multi = mocker.patch.object(TUI, "render_multi_status")

    a = ("a", "slurm", _st())
    b = ("b", "lepton", _st())
    items = [a, b]
    console = Console()

    TUI.render_status(items, console=console)

    mock_swarm.assert_not_called()
    mock_multi.assert_called_once()
    # First positional arg is the items list
    passed_items = mock_multi.call_args.args[0]
    assert isinstance(passed_items, list)
    assert passed_items == items
    # console is forwarded as kwarg
    assert mock_multi.call_args.kwargs.get("console") is console


def test_render_status_with_generator_is_materialized_and_passed_to_multi(mocker):
    mock_multi = mocker.patch.object(TUI, "render_multi_status")

    items = [
        ("one", "slurm", _st()),
        ("two", "lepton", _st()),
        ("three", "slurm", _st()),
    ]
    gen = (x for x in items)

    TUI.render_status(gen, console=None)

    # Ensure generator was consumed to a list and forwarded
    passed_items = mock_multi.call_args.args[0]
    assert isinstance(passed_items, list)
    assert passed_items == items
    # console explicitly forwarded as None
    assert "console" in mock_multi.call_args.kwargs
    assert mock_multi.call_args.kwargs["console"] is None


def test_render_status_forwards_console_none_explicitly_to_single(mocker):
    mock_swarm = mocker.patch.object(TUI, "render_swarm_status")
    status = _st()

    TUI.render_status(("n", "slurm", status), console=None)

    assert mock_swarm.call_args.kwargs.get("console") is None
