import json
from pathlib import Path

from typer.testing import CliRunner

# Adjust this import to where your Typer app lives
import domyn_swarm.cli.job as mod

runner = CliRunner()


# ---------------------------
# Helpers
# ---------------------------


def _mk_files(tmp_path: Path):
    in_path = tmp_path / "input.parquet"
    out_path = tmp_path / "output.parquet"
    in_path.write_bytes(b"PARQUET_MOCK")
    return in_path, out_path


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
    mocker.patch.object(mod.DomynLLMSwarm, "from_state", return_value=swarm)

    result = runner.invoke(
        mod.job_app,
        ["submit-script", str(script), "-n", "my-swarm", "--", "X", "Y"],
    )

    assert result.exit_code == 0
    swarm.submit_script.assert_called_once()
    assert swarm.submit_script.call_args.kwargs["extra_args"] == ["X", "Y"]


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
    cm = mocker.MagicMock()
    cm.__enter__.return_value = swarm
    cm.__exit__.return_value = None
    mocker.patch.object(mod, "DomynLLMSwarm", return_value=cm)

    # Mock _load_job to return a job object and assert some inputs
    job_obj = object()

    def _fake_load_job(job_class, job_kwargs, **kwargs):
        assert job_class == "domyn_swarm.jobs:ChatCompletionJob"
        assert isinstance(job_kwargs, str)
        assert kwargs["endpoint"] == swarm.endpoint
        assert kwargs["model"] == swarm.model
        # Ensure the kwargs JSON is parseable
        _ = json.loads(job_kwargs)
        return job_obj

    mocker.patch.object(mod, "_load_job", side_effect=_fake_load_job)

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


def test_submit_job_with_name_happy_path(mocker, tmp_path: Path):
    in_path, out_path = _mk_files(tmp_path)

    swarm = mocker.MagicMock()
    swarm.endpoint = "http://endpoint"
    swarm.model = "m"
    mocker.patch.object(mod.DomynLLMSwarm, "from_state", return_value=swarm)

    job_obj = object()
    mocker.patch.object(mod, "_load_job", return_value=job_obj)

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

    # Raise KeyboardInterrupt from _load_job
    def _boom(*a, **k):
        raise KeyboardInterrupt()

    mocker.patch.object(mod, "_load_job", side_effect=_boom)

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

    mocker.patch.object(mod, "_load_job", side_effect=KeyboardInterrupt)
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
    mocker.patch.object(mod, "_load_job", return_value=job_obj)

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

    assert res.exit_code == 0
    swarm.submit_job.assert_called_once()
    kwargs = swarm.submit_job.call_args.kwargs
    assert kwargs["num_threads"] == 3
    assert kwargs["limit"] == 50
    assert kwargs["detach"] is True
    assert kwargs["mail_user"] == "me@example.com"
    assert kwargs["checkpoint_dir"] == tmp_path / ".ckpt"
