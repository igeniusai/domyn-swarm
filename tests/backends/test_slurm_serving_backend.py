from types import SimpleNamespace

import pytest

import domyn_swarm.backends.serving.slurm as mod
from domyn_swarm.backends.serving.slurm import SlurmServingBackend
from domyn_swarm.platform.protocols import ServingHandle


class FakeDriver:
    def __init__(self, submit_jobid=111, submit_lb_jobid=222):
        self.submit_jobid = submit_jobid
        self.submit_lb_jobid = submit_lb_jobid
        self.calls = {"submit_replicas": None, "submit_endpoint": None}

    def submit_replicas(
        self, name, replicas, nodes, gpus_per_node, gpus_per_replica, replicas_per_node
    ):
        self.calls["submit_replicas"] = {
            "name": name,
            "replicas": replicas,
            "nodes": nodes,
            "gpus_per_node": gpus_per_node,
            "gpus_per_replica": gpus_per_replica,
            "replicas_per_node": replicas_per_node,
        }
        return self.submit_jobid

    def submit_endpoint(self, name, jobid, replicas):
        self.calls["submit_endpoint"] = {
            "name": name,
            "jobid": jobid,
            "replicas": replicas,
        }
        return self.submit_lb_jobid


def mk_cfg(port=9000, poll=10):
    endpoint = SimpleNamespace(port=port, poll_interval=poll)
    return SimpleNamespace(endpoint=endpoint)


# -------------------
# Tests
# -------------------


def test_create_or_update_calls_driver_and_returns_handle():
    driver = FakeDriver(submit_jobid=101, submit_lb_jobid=202)
    cfg = mk_cfg(port=9100, poll=7)
    be = SlurmServingBackend(driver=driver, cfg=cfg)

    spec = {
        "replicas": 2,
        "nodes": 3,
        "gpus_per_node": 4,
        "gpus_per_replica": 2,
        "replicas_per_node": 2,
    }

    handle = be.create_or_update("my-swarm", spec)

    # Driver calls captured
    assert driver.calls["submit_replicas"] == {
        "name": "my-swarm",
        "replicas": 2,
        "nodes": 3,
        "gpus_per_node": 4,
        "gpus_per_replica": 2,
        "replicas_per_node": 2,
    }
    assert driver.calls["submit_endpoint"] == {
        "name": "my-swarm",
        "jobid": 101,
        "replicas": 2,
    }

    # Handle returned
    assert isinstance(handle, ServingHandle)
    assert handle.id == "202"  # lb_jobid as string
    assert handle.url == ""
    assert handle.meta["jobid"] == 101
    assert handle.meta["lb_jobid"] == 202
    assert handle.meta["port"] == 9100
    assert handle.meta["name"] == "my-swarm"


def test_wait_ready_uses_injected_readiness(monkeypatch):
    class FakeProbe:
        def __init__(self):
            self.calls = []

        def wait_ready(self, handle, timeout_s):
            self.calls.append((handle, timeout_s))
            handle.url = "http://node:9000"
            handle.meta["lb_node"] = "node"
            return handle

    driver = FakeDriver()
    cfg = mk_cfg(port=9000, poll=5)
    probe = FakeProbe()
    be = SlurmServingBackend(driver=driver, cfg=cfg, readiness=probe)

    handle = ServingHandle(id="202", url="", meta={"jobid": 101, "lb_jobid": 202})
    out = be.wait_ready(handle, timeout_s=30)

    assert out.url == "http://node:9000"
    assert probe.calls == [(handle, 30)]  # used injected probe


def test_wait_ready_constructs_probe_with_cfg_values(monkeypatch):
    constructed = {}

    class FakeProbeCtor:
        def __init__(self, driver, endpoint_port, poll_interval_s):
            constructed["driver"] = driver
            constructed["endpoint_port"] = endpoint_port
            constructed["poll_interval_s"] = poll_interval_s

        def wait_ready(self, handle, timeout_s):
            handle.url = f"http://lb:{constructed['endpoint_port']}"
            handle.meta["lb_node"] = "lb"
            return handle

    # Patch the symbol used by the module
    monkeypatch.setattr(mod, "SlurmReadiness", FakeProbeCtor)

    driver = FakeDriver()
    cfg = mk_cfg(port=8123, poll=1.5)
    be = SlurmServingBackend(driver=driver, cfg=cfg)  # no readiness injected

    handle = ServingHandle(id="202", url="", meta={"jobid": 101, "lb_jobid": 202})
    out = be.wait_ready(handle, timeout_s=5)

    assert constructed["driver"] is driver
    assert constructed["endpoint_port"] == 8123
    assert constructed["poll_interval_s"] == 1.5
    assert out.url == "http://lb:8123"
    assert out.meta["lb_node"] == "lb"


def test_delete_scancels_both(monkeypatch):
    calls = []

    def fake_run(argv, check=False):
        calls.append((tuple(argv), check))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(mod, "subprocess", SimpleNamespace(run=fake_run))

    driver = FakeDriver()
    cfg = mk_cfg()
    be = SlurmServingBackend(driver=driver, cfg=cfg)

    h = ServingHandle(id="202", url="", meta={"jobid": 101, "lb_jobid": 202})
    be.delete(h)

    assert calls == [
        (("scancel", "101"), False),
        (("scancel", "202"), False),
    ]


def test_delete_ignores_missing_ids(monkeypatch):
    calls = []

    def fake_run(argv, check=False):
        calls.append((tuple(argv), check))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(mod, "subprocess", SimpleNamespace(run=fake_run))

    be = SlurmServingBackend(driver=FakeDriver(), cfg=mk_cfg())

    # Only lb_jobid present
    h = ServingHandle(id="x", url="", meta={"lb_jobid": 555})
    be.delete(h)

    assert calls == [(("scancel", "555"), False)]


def test_ensure_ready_raises_when_handle_none():
    be = SlurmServingBackend(driver=FakeDriver(), cfg=mk_cfg())
    with pytest.raises(RuntimeError, match="Serving handle is null"):
        be.ensure_ready(None)


@pytest.mark.parametrize(
    "meta,url_ok",
    [
        ({"jobid": 1, "lb_jobid": 2, "lb_node": None}, True),
        ({"jobid": 1, "lb_jobid": None, "lb_node": "n"}, True),
        ({"jobid": None, "lb_jobid": 2, "lb_node": "n"}, True),
        ({}, True),
    ],
)
def test_ensure_ready_raises_when_missing_fields(meta, url_ok):
    be = SlurmServingBackend(driver=FakeDriver(), cfg=mk_cfg())
    handle = ServingHandle(id="h", url=("ok" if url_ok else ""), meta=meta)
    with pytest.raises(RuntimeError, match="Swarm not ready"):
        be.ensure_ready(handle)


def test_ensure_ready_ok():
    be = SlurmServingBackend(driver=FakeDriver(), cfg=mk_cfg())
    handle = ServingHandle(
        id="h",
        url="http://n:9000",
        meta={"jobid": 1, "lb_jobid": 2, "lb_node": "node"},
    )
    # Should not raise
    be.ensure_ready(handle)
