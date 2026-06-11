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

import pytest
import typer

from domyn_swarm.cli import monitor as monitor_mod
from domyn_swarm.cli.monitor import build_prometheus_url, resolve_grafatui_argv


def _swarm(endpoint="http://lbnode:9000", enabled=True, route="/prometheus"):
    mon = SimpleNamespace(enabled=enabled, route_prefix=route)
    ep = SimpleNamespace(monitoring=mon)
    backend = SimpleNamespace(endpoint=ep)
    cfg = SimpleNamespace(backend=backend, name="my-swarm", model="Qwen/Qwen3-32B", replicas=16)
    return SimpleNamespace(endpoint=endpoint, cfg=cfg)


def test_build_prometheus_url_joins_route_prefix():
    assert build_prometheus_url(_swarm()) == "http://lbnode:9000/prometheus"


def test_build_prometheus_url_strips_trailing_slash():
    assert (
        build_prometheus_url(_swarm(endpoint="http://lbnode:9000/"))
        == "http://lbnode:9000/prometheus"
    )


def test_resolve_argv_includes_dashboard(tmp_path):
    dash = tmp_path / "vllm.json"
    dash.write_text("{}")
    argv = resolve_grafatui_argv("http://x/prometheus", dashboard=dash, extra=["--range", "1h"])
    assert argv[0] == "grafatui"
    assert "--prometheus-url" in argv and "http://x/prometheus" in argv
    assert "--grafana-json" in argv and str(dash) in argv
    assert "--range" in argv and "1h" in argv


def test_resolve_argv_without_dashboard():
    argv = resolve_grafatui_argv("http://x/prometheus", dashboard=None, extra=[])
    assert "--grafana-json" not in argv


def test_monitor_uses_custom_dashboard(monkeypatch, tmp_path):
    swarm = _swarm()
    custom = tmp_path / "custom.json"
    custom.write_text("{}")

    from domyn_swarm.core.state.state_manager import SwarmStateManager

    monkeypatch.setattr(SwarmStateManager, "load", classmethod(lambda cls, deployment_name: swarm))
    monkeypatch.setattr(monitor_mod.shutil, "which", lambda _: "/usr/bin/grafatui")
    captured: dict = {}
    monkeypatch.setattr(monitor_mod.os, "execvp", lambda f, argv: captured.update(argv=argv))

    monitor_mod.monitor("some-swarm", dashboard=custom)

    assert "--grafana-json" in captured["argv"]
    assert str(custom) in captured["argv"]


def test_monitor_rejects_missing_custom_dashboard(monkeypatch, tmp_path):
    swarm = _swarm()

    from domyn_swarm.core.state.state_manager import SwarmStateManager

    monkeypatch.setattr(SwarmStateManager, "load", classmethod(lambda cls, deployment_name: swarm))
    monkeypatch.setattr(monitor_mod.shutil, "which", lambda _: "/usr/bin/grafatui")

    with pytest.raises(typer.Exit) as ei:
        monitor_mod.monitor("some-swarm", dashboard=tmp_path / "missing.json")
    assert ei.value.exit_code == 2


def test_resolve_argv_emits_variables():
    argv = resolve_grafatui_argv(
        "http://x/prometheus",
        dashboard=None,
        extra=[],
        variables={"swarm": "my-swarm", "replicas": "16"},
    )
    assert argv.count("--var") == 2
    assert "swarm=my-swarm" in argv
    assert "replicas=16" in argv


def test_monitor_autofills_and_overrides_variables(monkeypatch):
    swarm = _swarm()

    from domyn_swarm.core.state.state_manager import SwarmStateManager

    monkeypatch.setattr(SwarmStateManager, "load", classmethod(lambda cls, deployment_name: swarm))
    monkeypatch.setattr(monitor_mod.shutil, "which", lambda _: "/usr/bin/grafatui")
    captured: dict = {}
    monkeypatch.setattr(monitor_mod.os, "execvp", lambda f, argv: captured.update(argv=argv))

    # User overrides model; swarm/vllm_job/replicas come from config.
    monitor_mod.monitor("some-swarm", var=["model=custom-model"])

    argv = captured["argv"]
    assert "swarm=my-swarm" in argv
    assert "vllm_job=vllm" in argv
    assert "replicas=16" in argv
    assert "model=custom-model" in argv  # override wins over cfg.model
    assert "model=Qwen/Qwen3-32B" not in argv


def test_monitor_exits_cleanly_when_endpoint_has_no_monitoring(monkeypatch):
    # Simulate a non-Slurm swarm whose endpoint lacks a `monitoring` attribute.
    endpoint = SimpleNamespace()  # no `monitoring`
    backend = SimpleNamespace(endpoint=endpoint)
    swarm = SimpleNamespace(endpoint="http://x:9000", cfg=SimpleNamespace(backend=backend))

    from domyn_swarm.core.state.state_manager import SwarmStateManager

    monkeypatch.setattr(SwarmStateManager, "load", classmethod(lambda cls, deployment_name: swarm))

    with pytest.raises(typer.Exit) as ei:
        monitor_mod.monitor("some-swarm")
    assert ei.value.exit_code == 1
