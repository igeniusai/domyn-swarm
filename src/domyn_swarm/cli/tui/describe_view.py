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
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rich.box import HEAVY
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text
import yaml

from .tables import _kv_table


def _yaml_block(data: Mapping[str, Any] | None) -> Syntax:
    if not isinstance(data, Mapping) or not data:
        return Syntax("# (no config found in state)", "yaml", word_wrap=True)
    return Syntax(yaml.safe_dump(data, sort_keys=False), "yaml", word_wrap=True)


def render_swarm_description(
    *,
    name: str,
    backend: str,
    cfg: Mapping[str, Any] | None,
    endpoint: str | None = None,
    console: Console | None = None,
) -> None:
    """
    Render a full-width description panel for a single swarm (no live status).

    Sections:
      • Overview (name, backend, endpoint, model basics)
      • Configuration (pretty YAML of stored config)
    """
    console = console or Console()

    # Banner
    banner = Text.assemble(
        ("SWARM ", "bold dim"),
        (name, "bold bright_cyan"),
        ("  •  ", "dim"),
        (backend.upper(), "bold magenta"),
    )

    # Overview / metadata (best-effort reads from cfg)
    meta = _kv_table()
    meta.add_row("Name", Text(name, style="bold cyan"))
    meta.add_row("Backend", backend.upper())

    if endpoint:
        url_txt = Text(endpoint)
        url_txt.stylize(f"link {endpoint}")
        meta.add_row("Endpoint", url_txt)

    # Pull a few common top-level fields if present
    model = isinstance(cfg, Mapping) and cfg.get("model")
    replicas = isinstance(cfg, Mapping) and cfg.get("replicas")
    gpus_per_replica = isinstance(cfg, Mapping) and cfg.get("gpus_per_replica")
    port = isinstance(cfg, Mapping) and cfg.get("port")
    image = isinstance(cfg, Mapping) and (
        cfg.get("image") or (cfg.get("backend", {}) or {}).get("endpoint", {}).get("image")
    )

    if model:
        meta.add_row("Model", str(model))
    if replicas is not None:
        meta.add_row("Replicas", str(replicas))
    if gpus_per_replica is not None:
        meta.add_row("GPUs/replica", str(gpus_per_replica))
    if port is not None:
        meta.add_row("Port", str(port))
    if image:
        meta.add_row("Image", str(image))

    # Body group
    body = Group(
        Rule(banner, style="cyan"),
        meta,
        Panel(
            _yaml_block(cfg),
            title="Configuration (from state)",
            border_style="dim",
            padding=(1, 2),
            box=HEAVY,
        ),
    )

    console.print(
        Panel(
            body,
            title=f"[b cyan]{name}[/] — [magenta]{backend.upper()}[/]",
            border_style="cyan",
            padding=(1, 2),
            box=HEAVY,
            expand=True,
        )
    )
