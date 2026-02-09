import json
from pathlib import Path

from typer.testing import CliRunner

# Adjust this import to where your Typer app lives
import domyn_swarm.cli.job as mod
from domyn_swarm.platform.protocols import JobHandle, JobStatus

runner = CliRunner()


# ---------------------------
# Helpers
# ---------------------------


def _mk_files(tmp_path: Path):
    in_path = tmp_path / "input.parquet"
    out_path = tmp_path / "output.parquet"
    in_path.write_bytes(b"PARQUET_MOCK")
    return in_path, out_path


def _parse_last_json_line(text: str) -> dict:
    lines = [line for line in text.splitlines() if line.strip()]
    assert lines, "No CLI output lines found."
    return json.loads(lines[-1])


# ---------------------------
# submit-script
# ---------------------------


def test_submit_script_with_config_runs_in_context(mocker, tmp_path: Path):
    script = tmp_path / "script.py"
    script.write_text("print('hi')")

    config_path = tmp_path / "config.yaml"
    config_path.write_text("model: test-model\nname: test-swarm")

    cfg_obj = object()
    mocker.patch.object(mod, "_load_swarm_config", return_value=cfg_obj)

    # Mock DomynLLMSwarm to return a context manager whose __enter__ returns a swarm
    swarm = mocker.MagicMock()
    cm = mocker.MagicMock()
    cm.__enter__.return_value = swarm
    cm.__exit__.return_value = None
    mocker.patch.object(mod, "DomynLLMSwarm", return_value=cm)

    result = runner.invoke(
        mod.job_app,
        [
            "submit-script",
            str(script),
            "-c",
            str(tmp_path / "config.yaml"),
            "--",
            "foo",
            "bar",
        ],
    )

    print(result.stderr)

    assert result.exit_code == 0
    swarm.submit_script.assert_called_once()
    assert swarm.submit_script.call_args.kwargs["extra_args"] == ["foo", "bar"]


def test_submit_script_with_name_uses_from_state(mocker, tmp_path: Path):
    script = tmp_path / "script.py"
    script.write_text("print('hi')")

    swarm = mocker.MagicMock()
    swarm.name = "my-swarm"
    swarm.submit_script.return_value = JobHandle(
        id="123.0",
        status=JobStatus.RUNNING,
        meta={"job_id": "job-1", "pid": 4321, "external_id": "123.0"},
    )
    mocker.patch.object(mod.DomynLLMSwarm, "from_state", return_value=swarm)

    result = runner.invoke(
        mod.job_app,
        ["submit-script", str(script), "-n", "my-swarm", "--", "X", "Y"],
    )

    assert result.exit_code == 0
    swarm.submit_script.assert_called_once()
    assert swarm.submit_script.call_args.kwargs["extra_args"] == ["X", "Y"]
    payload = _parse_last_json_line(result.stdout)
    assert payload["command"] == "submit-script"
    assert payload["swarm"] == "my-swarm"
    assert payload["job_id"] == "job-1"
    assert payload["pid"] == 4321
    assert payload["external_id"] == "123.0"


def test_submit_script_mutual_exclusion(mocker, tmp_path: Path):
    script = tmp_path / "script.py"
    script.write_text("print('hi')")

    result = runner.invoke(
        mod.job_app,
        [
            "submit-script",
            str(script),
            "-c",
            str(tmp_path / "config.yaml"),
            "-n",
            "my-swarm",
        ],
    )
    # Should error because both --config and --name were provided
    assert result.exit_code != 0


def test_submit_script_raises_without_config_or_name(mocker, tmp_path: Path):
    script = tmp_path / "script.py"
    script.write_text("print('hi')")

    result = runner.invoke(mod.job_app, ["submit-script", str(script)])
    assert result.exit_code != 0
    assert "State is null" in str(result.exception)


# ---------------------------
# submit (job)
# ---------------------------


def test_submit_job_with_config_happy_path(mocker, tmp_path: Path):
    in_path, out_path = _mk_files(tmp_path)
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("model: my-model\nname: my-swarm")

    # Mock config loader
    cfg_obj = object()
    mocker.patch.object(mod, "_load_swarm_config", return_value=cfg_obj)

    # Mock swarm context manager
    swarm = mocker.MagicMock()
    swarm.endpoint = "http://host:9000"
    swarm.model = "my-model"
    swarm.name = "cfg-swarm"
    swarm.submit_job.return_value = JobHandle(
        id="123.0",
        status=JobStatus.PENDING,
        meta={"job_id": "job-1", "pid": 4321, "external_id": "123.0"},
    )
    cm = mocker.MagicMock()
    cm.__enter__.return_value = swarm
    cm.__exit__.return_value = None
    mocker.patch.object(mod, "DomynLLMSwarm", return_value=cm)

    # Mock JobBuilder to return a job object and assert some inputs
    job_obj = object()

    def _fake_build(job_class, job_kwargs, **kwargs):
        assert job_class == "domyn_swarm.jobs:ChatCompletionJob"
        assert isinstance(job_kwargs, str)
        assert kwargs["endpoint"] == swarm.endpoint
        assert kwargs["model"] == swarm.model
        # Ensure the kwargs JSON is parseable
        _ = json.loads(job_kwargs)
        return job_obj

    mocker.patch.object(mod.helpers.JobBuilder, "from_class_path", side_effect=_fake_build)

    res = runner.invoke(
        mod.job_app,
        [
            "submit",
            "--input",
            str(in_path),
            "--output",
            str(out_path),
            "-c",
            str(tmp_path / "cfg.yaml"),
            "--job-kwargs",
            '{"temperature":0.2}',
            "--checkpoint-interval",
            "16",
            "--max-concurrency",
            "8",
            "--retries",
            "2",
            "--timeout",
            "123",
            "--num-threads",
            "4",
            "--limit",
            "100",
            "--detach",
        ],
    )
    print(res.stderr)
    assert res.exit_code == 0

    swarm.submit_job.assert_called_once()
    call = swarm.submit_job.call_args
    assert call.args[0] is job_obj
    k = call.kwargs
    assert k["input_path"] == in_path
    assert k["output_path"] == out_path
    assert k["num_threads"] == 4
    assert k["limit"] == 100
    assert k["detach"] is True
    assert "checkpoint_dir" in k
    payload = _parse_last_json_line(res.stdout)
    assert payload["command"] == "submit"
    assert payload["swarm"] == "cfg-swarm"
    assert payload["job_id"] == "job-1"
    assert payload["pid"] == 4321
    assert payload["external_id"] == "123.0"


def test_submit_job_with_name_happy_path(mocker, tmp_path: Path):
    in_path, out_path = _mk_files(tmp_path)

    swarm = mocker.MagicMock()
    swarm.endpoint = "http://endpoint"
    swarm.model = "m"
    mocker.patch.object(mod.DomynLLMSwarm, "from_state", return_value=swarm)

    job_obj = object()
    mocker.patch.object(mod.helpers.JobBuilder, "from_class_path", return_value=job_obj)

    res = runner.invoke(
        mod.job_app,
        [
            "submit",
            "--input",
            str(in_path),
            "--output",
            str(out_path),
            "-n",
            "my-swarm",
        ],
    )

    assert res.exit_code == 0
    swarm.submit_job.assert_called_once()
    assert swarm.submit_job.call_args.args[0] is job_obj


def test_submit_job_mutual_exclusion(mocker, tmp_path: Path):
    in_path, out_path = _mk_files(tmp_path)

    res = runner.invoke(
        mod.job_app,
        [
            "submit",
            "--input",
            str(in_path),
            "--output",
            str(out_path),
            "-c",
            str(tmp_path / "x.yaml"),
            "-n",
            "name",
        ],
    )
    assert res.exit_code != 0  # typer.Exit(1)


def test_submit_job_raises_if_no_config_and_no_name(mocker, tmp_path: Path):
    in_path, out_path = _mk_files(tmp_path)
    res = runner.invoke(
        mod.job_app,
        ["submit", "--input", str(in_path), "--output", str(out_path)],
    )
    assert res.exit_code != 0
    assert "Swarm name is null" in str(res.exception)


def test_submit_job_keyboard_interrupt_abort(mocker, tmp_path: Path):
    in_path, out_path = _mk_files(tmp_path)
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("model: my-model\nname: my-swarm")

    cfg_obj = object()
    mocker.patch.object(mod, "_load_swarm_config", return_value=cfg_obj)

    # Context manager
    swarm = mocker.MagicMock()
    cm = mocker.MagicMock()
    cm.__enter__.return_value = swarm
    cm.__exit__.return_value = None
    cm.cleanup = mocker.MagicMock()
    mocker.patch.object(mod, "DomynLLMSwarm", return_value=cm)

    # Raise KeyboardInterrupt from job builder
    def _boom(*a, **k):
        raise KeyboardInterrupt()

    mocker.patch.object(mod.helpers.JobBuilder, "from_class_path", side_effect=_boom)

    # User chooses to abort → True; expect cleanup and abort
    mocker.patch.object(mod.typer, "confirm", return_value=True)

    res = runner.invoke(
        mod.job_app,
        [
            "submit",
            "--input",
            str(in_path),
            "--output",
            str(out_path),
            "-c",
            str(tmp_path / "cfg.yaml"),
        ],
    )
    assert res.exit_code != 0  # Aborted
    print(res.stdout)
    print(res.stderr)
    cm.cleanup.assert_called_once()
    assert "Swarm allocation cancelled by user" in res.stdout


def test_submit_job_keyboard_interrupt_continue(mocker, tmp_path: Path):
    in_path, out_path = _mk_files(tmp_path)
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("model: my-model\nname: my-swarm")

    cfg_obj = object()
    mocker.patch.object(mod, "_load_swarm_config", return_value=cfg_obj)

    swarm = mocker.MagicMock()
    cm = mocker.MagicMock()
    cm.__enter__.return_value = swarm
    cm.__exit__.return_value = None
    cm.cleanup = mocker.MagicMock()
    mocker.patch.object(mod, "DomynLLMSwarm", return_value=cm)

    mocker.patch.object(mod.helpers.JobBuilder, "from_class_path", side_effect=KeyboardInterrupt)
    mocker.patch.object(mod.typer, "confirm", return_value=False)

    res = runner.invoke(
        mod.job_app,
        [
            "submit",
            "--input",
            str(in_path),
            "--output",
            str(out_path),
            "-c",
            str(tmp_path / "cfg.yaml"),
        ],
    )
    assert res.exit_code == 0
    cm.cleanup.assert_not_called()
    # Avoid being strict with ellipsis char
    assert "Continuing to wait for job to complete" in res.stdout


def test_submit_job_forwards_specific_options(mocker, tmp_path: Path):
    in_path, out_path = _mk_files(tmp_path)
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("model: my-model\nname: my-swarm")

    cfg_obj = object()
    mocker.patch.object(mod, "_load_swarm_config", return_value=cfg_obj)

    swarm = mocker.MagicMock()
    swarm.endpoint = "http://host:9000"
    swarm.model = "m"
    cm = mocker.MagicMock()
    cm.__enter__.return_value = swarm
    cm.__exit__.return_value = None
    mocker.patch.object(mod, "DomynLLMSwarm", return_value=cm)

    job_obj = object()
    mocker.patch.object(mod.helpers.JobBuilder, "from_class_path", return_value=job_obj)

    res = runner.invoke(
        mod.job_app,
        [
            "submit",
            "--input",
            str(in_path),
            "--output",
            str(out_path),
            "-c",
            str(tmp_path / "cfg.yaml"),
            "--num-threads",
            "3",
            "--limit",
            "50",
            "--detach",
            "--mail-user",
            "me@example.com",
            "--checkpoint-dir",
            str(tmp_path / ".ckpt"),
        ],
    )

    print(res.output)

    assert res.exit_code == 0
    swarm.submit_job.assert_called_once()
    kwargs = swarm.submit_job.call_args.kwargs
    assert kwargs["num_threads"] == 3
    assert kwargs["limit"] == 50
    assert kwargs["detach"] is True
    assert kwargs["mail_user"] == "me@example.com"
    assert kwargs["checkpoint_dir"] == tmp_path / ".ckpt"


def test_submit_job_forwards_ray_address(mocker, tmp_path: Path):
    in_path, out_path = _mk_files(tmp_path)
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("model: my-model\nname: my-swarm")

    cfg_obj = object()
    mocker.patch.object(mod, "_load_swarm_config", return_value=cfg_obj)

    swarm = mocker.MagicMock()
    swarm.endpoint = "http://host:9000"
    swarm.model = "m"
    cm = mocker.MagicMock()
    cm.__enter__.return_value = swarm
    cm.__exit__.return_value = None
    mocker.patch.object(mod, "DomynLLMSwarm", return_value=cm)

    job_obj = object()
    mocker.patch.object(mod.helpers.JobBuilder, "from_class_path", return_value=job_obj)

    res = runner.invoke(
        mod.job_app,
        [
            "submit",
            "--input",
            str(in_path),
            "--output",
            str(out_path),
            "-c",
            str(config_path),
            "--data-backend",
            "ray",
            "--id-column",
            "doc_id",
            "--ray-address",
            "ray://head:10001",
        ],
    )

    assert res.exit_code == 0
    swarm.submit_job.assert_called_once()
    kwargs = swarm.submit_job.call_args.kwargs
    assert kwargs["ray_address"] == "ray://head:10001"


def test_wait_job_with_job_id_updates_status_and_emits_json(mocker):
    handle = JobHandle(id="123.0", status=JobStatus.RUNNING, meta={"job_id": "job-1"})
    target = mod.helpers.ResolvedJobTarget(
        swarm_name="my-swarm",
        handle=handle,
        job_id="job-1",
        source="job_id",
    )
    resolve = mocker.patch.object(mod.helpers, "resolve_job_target", return_value=target)
    emit = mocker.patch.object(mod.helpers, "emit_job_control_json")
    update = mocker.patch.object(mod.SwarmStateManager, "update_job")

    swarm = mocker.MagicMock()
    swarm.wait_job.return_value = JobStatus.SUCCEEDED
    mocker.patch.object(mod.DomynLLMSwarm, "from_state", return_value=swarm)

    result = runner.invoke(mod.job_app, ["wait", "--job-id", "job-1"])

    assert result.exit_code == 0
    resolve.assert_called_once_with(
        job_id="job-1",
        external_id=None,
        handle_json=None,
        deployment_name=None,
    )
    swarm.wait_job.assert_called_once_with(handle, stream_logs=True)
    update.assert_called_once_with(
        "job-1",
        status=JobStatus.SUCCEEDED,
        external_id=None,
    )
    emit.assert_called_once()
    assert target.handle.status == JobStatus.SUCCEEDED


def test_cancel_job_with_external_id_sets_cancelled_and_updates_state(mocker):
    handle = JobHandle(
        id="123.0",
        status=JobStatus.RUNNING,
        meta={"job_id": "job-1", "external_id": "123.0"},
    )
    target = mod.helpers.ResolvedJobTarget(
        swarm_name="my-swarm",
        handle=handle,
        job_id="job-1",
        source="external_id",
    )
    resolve = mocker.patch.object(mod.helpers, "resolve_job_target", return_value=target)
    emit = mocker.patch.object(mod.helpers, "emit_job_control_json")
    update = mocker.patch.object(mod.SwarmStateManager, "update_job")

    swarm = mocker.MagicMock()
    swarm.cancel_job.return_value = JobStatus.CANCELLED
    mocker.patch.object(mod.DomynLLMSwarm, "from_state", return_value=swarm)

    result = runner.invoke(
        mod.job_app,
        ["cancel", "--external-id", "123.0", "--name", "my-swarm"],
    )

    assert result.exit_code == 0
    resolve.assert_called_once_with(
        job_id=None,
        external_id="123.0",
        handle_json=None,
        deployment_name="my-swarm",
    )
    swarm.cancel_job.assert_called_once_with(handle)
    update.assert_called_once_with(
        "job-1",
        status=JobStatus.CANCELLED,
        external_id="123.0",
    )
    emit.assert_called_once()
    assert target.handle.status == JobStatus.CANCELLED
