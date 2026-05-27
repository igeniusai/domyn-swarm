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

from collections.abc import Iterable, Mapping
import json
import shlex
from typing import Any

from rich.console import Console
from rich.padding import Padding
from rich.syntax import Syntax
from rich.text import Text

from .badges import phase_badge
from .tables import _kv_table, list_table


def _status_badge(status: object) -> Text:
    """Render a styled status badge.

    Args:
        status: Raw status value.

    Returns:
        Rich text badge for the status.
    """
    status_text = str(status or "UNKNOWN").upper()
    return phase_badge(status_text)


def _fmt_command(command: object) -> str:
    """Format command payload for TUI display.

    Args:
        command: Command payload from state.

    Returns:
        Human-readable command representation.
    """
    if isinstance(command, list) and all(isinstance(part, str) for part in command):
        return shlex.join(command)
    if isinstance(command, str):
        return command
    return "—"


def render_job_list(
    rows: Iterable[Mapping[str, Any]], *, swarm_name: str, console: Console
) -> None:
    """Render a compact jobs table.

    Args:
        rows: Job record mappings from state.
        swarm_name: Swarm deployment name.
        console: Rich console used for output.
    """
    rows_list = list(rows)
    if not rows_list:
        console.print(f"[yellow]No jobs found for swarm '{swarm_name}'.[/]")
        return

    table = list_table(
        columns=[" Job ID", "Status", "Kind", "Provider", "External ID", "Updated", "Name"]
    )
    for row in rows_list:
        job_id = Padding(str(row.get("job_id") or "—"), (0, 0, 0, 1))
        status = _status_badge(row.get("status"))
        kind = str(row.get("kind") or "—")
        provider = str(row.get("provider") or "—")
        external_id = str(row.get("external_id") or "—")
        updated = str(row.get("update_dt") or row.get("creation_dt") or "—")
        name = str(row.get("name") or "—")
        table.add_row(job_id, status, kind, provider, external_id, updated, name)
    console.print(table)


def render_job_status(job: Mapping[str, Any], *, console: Console) -> None:
    """Render a detailed single-job status panel.

    Args:
        job: Job record mapping from state.
        console: Rich console used for output.
    """
    details = _kv_table()
    details.add_row("Job ID", Text(str(job.get("job_id") or "—"), style="bold cyan"))
    details.add_row("Swarm", str(job.get("deployment_name") or "—"))
    details.add_row("Status", _status_badge(job.get("status")))
    details.add_row("Provider", str(job.get("provider") or "—"))
    details.add_row("Kind", str(job.get("kind") or "—"))
    details.add_row("External ID", str(job.get("external_id") or "—"))
    details.add_row("Name", str(job.get("name") or "—"))
    details.add_row("Created", str(job.get("creation_dt") or "—"))
    details.add_row("Updated", str(job.get("update_dt") or "—"))
    details.add_row("Raw Status", str(job.get("raw_status") or "—"))
    details.add_row("Error", str(job.get("error") or "—"))
    if "refresh_source" in job:
        details.add_row("Refresh Source", str(job.get("refresh_source") or "—"))
    if "refresh_error" in job:
        details.add_row("Refresh Error", str(job.get("refresh_error") or "—"))
    details.add_row("Command", _fmt_command(job.get("command")))
    console.print(details)

    resources = job.get("resources")
    if isinstance(resources, dict) and resources:
        console.print(
            Syntax(json.dumps(resources, indent=2, sort_keys=True), "json", word_wrap=True)
        )

    log_paths = job.get("log_paths")
    if isinstance(log_paths, dict) and log_paths:
        console.print(
            Syntax(json.dumps(log_paths, indent=2, sort_keys=True), "json", word_wrap=True)
        )
