from importlib import resources
import json

SUPPORTED = {"timeseries", "stat", "gauge", "bargauge", "table", "heatmap", "graph"}


def _load():
    with resources.files("domyn_swarm.data.dashboards").joinpath("vllm.json").open() as fh:
        return json.load(fh)


def test_dashboard_is_valid_json_with_title_and_panels():
    d = _load()
    assert d["title"]
    assert isinstance(d["panels"], list) and d["panels"]


def test_every_panel_has_supported_type_and_targets():
    d = _load()
    for p in d["panels"]:
        assert p["type"] in SUPPORTED, p
        assert p["title"]
        assert "gridPos" in p
        assert p["targets"] and p["targets"][0]["expr"]
