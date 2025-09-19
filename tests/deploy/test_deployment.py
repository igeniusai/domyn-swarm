from types import SimpleNamespace

import pytest

from domyn_swarm.deploy.deployment import Deployment


def test_up_calls_serving_create_then_wait_and_stores_handle(mocker):
    serving = mocker.Mock()
    compute = mocker.Mock()
    extras = {"workspace": "ws-123"}

    h_created = SimpleNamespace(id="A", url="", meta={})
    h_ready = SimpleNamespace(id="B", url="http://ready", meta={})
    serving.create_or_update.return_value = h_created
    serving.wait_ready.return_value = h_ready

    dep = Deployment(serving=serving, compute=compute, extras=extras)

    # __enter__ returns self (context manager sugar)
    assert dep.__enter__() is dep

    out = dep.up("ep-name", {"model": "m1"}, timeout_s=60)

    serving.create_or_update.assert_called_once_with(
        "ep-name", {"model": "m1"}, extras=extras
    )
    serving.wait_ready.assert_called_once_with(h_created, 60, extras=extras)
    assert out is h_ready
    assert dep._handle is h_ready


def test_run_forwards_all_arguments_to_compute_with_extras(mocker):
    serving = mocker.Mock()
    compute = mocker.Mock()
    compute.submit.return_value = SimpleNamespace(
        id="job-1", status="PENDING", meta={"k": "v"}
    )
    extras = {"note": "hi"}

    dep = Deployment(serving=serving, compute=compute, extras=extras)

    out = dep.run(
        name="my-job",
        image="repo/img:tag",
        command=["python", "-m", "pkg.mod", "--arg", "v"],
        env={"ENDPOINT": "http://u", "MODEL": "m"},
        resources={"shape": "gpu.1x"},
        detach=True,
        nshards=4,
        shard_id=2,
    )

    compute.submit.assert_called_once_with(
        name="my-job",
        image="repo/img:tag",
        command=["python", "-m", "pkg.mod", "--arg", "v"],
        env={"ENDPOINT": "http://u", "MODEL": "m"},
        resources={"shape": "gpu.1x"},
        detach=True,
        nshards=4,
        shard_id=2,
        extras=extras,
    )
    assert out.id == "job-1"
    assert out.status == "PENDING"


def test_down_delegates_to_serving_delete(mocker):
    serving = mocker.Mock()
    compute = mocker.Mock()
    dep = Deployment(serving=serving, compute=compute)

    h = SimpleNamespace(id="ep1", url="http://x", meta={})
    dep.down(h)
    serving.delete.assert_called_once_with(h)


def test_context_manager_cleans_up_when_handle_set(mocker):
    serving = mocker.Mock()
    compute = mocker.Mock()
    dep = Deployment(serving=serving, compute=compute)

    h = SimpleNamespace(id="ep1", url="http://x", meta={})
    dep._handle = h

    with dep:
        # no-op inside; __exit__ should call delete
        pass

    serving.delete.assert_called_once_with(h)
    assert dep._handle is None


def test_context_manager_clears_handle_even_if_delete_raises(mocker):
    serving = mocker.Mock()
    compute = mocker.Mock()
    dep = Deployment(serving=serving, compute=compute)

    h = SimpleNamespace(id="ep2", url="http://y", meta={})
    dep._handle = h

    serving.delete.side_effect = RuntimeError("boom")

    # __exit__ does not swallow exceptions; ensure it still clears _handle
    with pytest.raises(RuntimeError, match="boom"):
        dep.__exit__(None, None, None)

    assert dep._handle is None


def test_ensure_ready_raises_without_handle(mocker):
    serving = mocker.Mock()
    compute = mocker.Mock()
    dep = Deployment(serving=serving, compute=compute)

    with pytest.raises(RuntimeError, match="No serving handle"):
        dep.ensure_ready()


def test_ensure_ready_delegates_to_serving(mocker):
    serving = mocker.Mock()
    compute = mocker.Mock()
    dep = Deployment(serving=serving, compute=compute)

    h = SimpleNamespace(id="ep3", url="http://ok", meta={})
    dep._handle = h
    serving.ensure_ready.return_value = "READY"

    out = dep.ensure_ready()
    serving.ensure_ready.assert_called_once_with(h)
    assert out == "READY"
