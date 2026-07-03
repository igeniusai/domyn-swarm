from pydantic import ValidationError
import pytest

from domyn_swarm.config.slurm import GpuExporterConfig, MonitoringConfig, SlurmEndpointConfig


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


def test_gpu_exporter_disabled_by_default():
    mon = MonitoringConfig(nginx_image="nginx.sif")
    assert mon.gpu_exporter.enabled is False
    assert mon.gpu_exporter.kind == "nvidia_smi"
    assert mon.gpu_exporter.port == 9835


def test_gpu_exporter_nvidia_smi_binary_default():
    gx = GpuExporterConfig(enabled=True, kind="nvidia_smi")
    assert gx.resolved_binary(mode="binary") == "nvidia_gpu_exporter"


def test_gpu_exporter_dcgm_image_default():
    gx = GpuExporterConfig(enabled=True, kind="dcgm")
    assert gx.resolved_image(mode="container") == (
        "nvcr.io/nvidia/k8s/dcgm-exporter:3.3.5-3.4.1-ubuntu22.04"
    )


def test_gpu_exporter_dcgm_binary_requires_explicit_binary():
    with pytest.raises(ValueError):
        GpuExporterConfig(enabled=True, kind="dcgm", binary=None).resolved_binary(mode="binary")


def test_dcgm_core_counters_bundled():
    from importlib import resources

    ref = resources.files("domyn_swarm.data.dcgm").joinpath("core-counters.csv")
    text = ref.read_text()
    assert "DCGM_FI_DEV_FB_USED" in text
    assert "DCGM_FI_PROF_" not in text  # profiling needs root; excluded on purpose


def test_nvidia_smi_container_mode_without_image_raises():
    with pytest.raises(ValueError):
        MonitoringConfig(
            mode="container",
            gpu_exporter=GpuExporterConfig(enabled=True, kind="nvidia_smi", image=None),
        )


def test_dcgm_binary_mode_raises():
    with pytest.raises(ValueError):
        MonitoringConfig(
            mode="binary",
            gpu_exporter=GpuExporterConfig(enabled=True, kind="dcgm"),
        )


def test_nvidia_smi_container_mode_with_explicit_image_does_not_raise():
    mon = MonitoringConfig(
        mode="container",
        gpu_exporter=GpuExporterConfig(enabled=True, kind="nvidia_smi", image="nvidia_smi.sif"),
    )
    assert mon.gpu_exporter.image == "nvidia_smi.sif"


def test_dcgm_container_mode_does_not_raise():
    mon = MonitoringConfig(
        mode="container",
        gpu_exporter=GpuExporterConfig(enabled=True, kind="dcgm"),
    )
    assert mon.gpu_exporter.kind == "dcgm"
