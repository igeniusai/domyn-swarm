import types

from domyn_swarm.backends.serving.slurm_driver import SlurmDriver


def test_get_job_state_prefers_squeue(monkeypatch):
    """Returns live state from squeue when available."""
    cfg = types.SimpleNamespace(backend=None)
    driver = SlurmDriver(cfg)

    def _run(cmd, check=False, text=True, capture_output=True):
        """Return a RUNNING state for squeue."""
        if cmd[0] == "squeue":
            return types.SimpleNamespace(returncode=0, stdout="RUNNING\n")
        return types.SimpleNamespace(returncode=1, stdout="")

    monkeypatch.setattr("subprocess.run", _run)
    assert driver.get_job_state(1) == "RUNNING"


def test_get_job_state_falls_back_to_sacct(monkeypatch):
    """Uses sacct when squeue is empty."""
    cfg = types.SimpleNamespace(backend=None)
    driver = SlurmDriver(cfg)

    def _run(cmd, check=False, text=True, capture_output=True):
        """Return a FAILED state for sacct."""
        if cmd[0] == "squeue":
            return types.SimpleNamespace(returncode=0, stdout="")
        if cmd[0] == "sacct":
            return types.SimpleNamespace(returncode=0, stdout="FAILED\n")
        return types.SimpleNamespace(returncode=1, stdout="")

    monkeypatch.setattr("subprocess.run", _run)
    assert driver.get_job_state(2) == "FAILED"


def test_get_job_state_falls_back_to_scontrol(monkeypatch):
    """Uses scontrol when squeue and sacct are empty."""
    cfg = types.SimpleNamespace(backend=None)
    driver = SlurmDriver(cfg)

    def _run(cmd, check=False, text=True, capture_output=True):
        """Return a COMPLETED state from scontrol."""
        if cmd[0] == "squeue":
            return types.SimpleNamespace(returncode=0, stdout="")
        if cmd[0] == "sacct":
            return types.SimpleNamespace(returncode=0, stdout="")
        if cmd[0] == "scontrol":
            return types.SimpleNamespace(returncode=0, stdout="JobState=COMPLETED Reason=None")
        return types.SimpleNamespace(returncode=1, stdout="")

    monkeypatch.setattr("subprocess.run", _run)
    assert driver.get_job_state(3) == "COMPLETED"


def test_get_job_state_unknown_when_all_missing(monkeypatch):
    """Returns UNKNOWN when no command yields a state."""
    cfg = types.SimpleNamespace(backend=None)
    driver = SlurmDriver(cfg)

    def _run(cmd, check=False, text=True, capture_output=True):
        """Return empty output for all commands."""
        return types.SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr("subprocess.run", _run)
    assert driver.get_job_state(4) == "UNKNOWN"
