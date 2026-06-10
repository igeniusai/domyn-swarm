from pydantic import ValidationError
import pytest

from domyn_swarm.config.slurm import MonitoringConfig, SlurmEndpointConfig


def test_monitoring_disabled_by_default():
    # nginx_image is normally sourced from defaults.yaml; pass it explicitly so the
    # test doesn't depend on a defaults file being present.
    ep = SlurmEndpointConfig(nginx_image="nginx.sif")
    assert ep.monitoring.enabled is False


def test_monitoring_defaults():
    m = MonitoringConfig()
    assert m.mode == "container"
    assert m.port == 9090
    assert m.route_prefix == "/prometheus"
    assert m.scrape_interval == "15s"
    assert m.retention == "12h"
    assert m.exporter_port == 9113
    assert m.prometheus_binary == "prometheus"
    assert m.nginx_exporter_binary == "nginx-prometheus-exporter"


def test_monitoring_route_prefix_normalized():
    assert MonitoringConfig(route_prefix="prometheus").route_prefix == "/prometheus"


def test_monitoring_mode_validated():
    with pytest.raises(ValidationError):
        MonitoringConfig(mode="rpm")


def test_monitoring_route_prefix_rejects_empty():
    with pytest.raises(ValidationError):
        MonitoringConfig(route_prefix="")
