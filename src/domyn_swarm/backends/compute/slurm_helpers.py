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

from collections.abc import Sequence
import contextlib
import os
from pathlib import Path
import re
import select
import shlex
import signal
import subprocess
import time

from domyn_swarm.platform.protocols import JobStatus

STEP_ID_STDOUT_TIMEOUT_S = 1.0
STEP_ID_POLL_TIMEOUT_S = 10.0
STEP_ID_POLL_INTERVAL_S = 0.5


def _normalize_returncode(returncode: int) -> JobStatus:
    """Normalize a `subprocess.Popen.returncode` to `JobStatus`."""
    if returncode == 0:
        return JobStatus.SUCCEEDED
    # Negative returncodes represent termination by signal.
    if returncode < 0:
        sig = -returncode
        if sig in (signal.SIGTERM, signal.SIGINT, signal.SIGKILL):
            return JobStatus.CANCELLED
        return JobStatus.FAILED
    return JobStatus.FAILED


def _pid_exists(pid: int) -> bool:
    """Return True if a PID appears to exist on this host."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stream_text_lines(src, dst) -> None:
    """Stream text lines from a readable file-like to a writable file-like."""
    try:
        for line in src:
            dst.write(line)
            dst.flush()
    except Exception:
        pass


def _terminate_process_group(pgid: int, *, grace_s: float = 10.0) -> None:
    """Terminate a POSIX process group, escalating to SIGKILL after a grace period."""
    os.killpg(pgid, signal.SIGTERM)
    deadline = time.time() + max(grace_s, 0.0)
    while time.time() < deadline:
        # If the group leader is gone, assume the group is gone.
        if not _pid_exists(pgid):
            return
        time.sleep(0.1)
    os.killpg(pgid, signal.SIGKILL)


def _wrap_with_step_echo(
    command: Sequence[str],
    step_id_fifo: Path | None,
) -> list[str]:
    """Wrap a command to emit the Slurm job/step id before exec.

    Args:
        command: The original command sequence.
        step_id_fifo: Optional FIFO path for step-id reporting.

    Returns:
        Wrapped command suitable for `srun`, e.g. `bash -lc 'echo ...; exec ...'`.
    """
    cmd_str = shlex.join([*map(str, command)])
    if step_id_fifo:
        fifo = shlex.quote(str(step_id_fifo))
        prefix = f'echo "${{SLURM_JOB_ID}}.${{SLURM_STEP_ID}}" > {fifo}'
    else:
        prefix = 'echo "${SLURM_JOB_ID}.${SLURM_STEP_ID}"'
    return ["bash", "-lc", f"{prefix}; exec {cmd_str}"]


def _resolve_external_id(
    *,
    stdout,
    job_id: int,
    step_name: str | None,
    step_id_fifo: Path | None,
) -> str | None:
    """Resolve Slurm external id from stdout or via Slurm tooling.

    Args:
        stdout: Process stdout stream.
        job_id: Parent Slurm job id.
        step_name: Step name to query in Slurm.
        step_id_fifo: Optional FIFO path for step-id reporting.

    Returns:
        Slurm job.step id if found, else None.
    """
    if step_id_fifo:
        external_id = _read_step_id_from_fifo(step_id_fifo, timeout_s=STEP_ID_STDOUT_TIMEOUT_S)
        if external_id:
            _try_cleanup_fifo(step_id_fifo)
            return external_id
    external_id = _read_step_id_from_stream(stdout, timeout_s=STEP_ID_STDOUT_TIMEOUT_S)
    if external_id:
        return external_id
    if not step_name:
        return None
    return _wait_for_step_id(
        job_id=job_id,
        step_name=step_name,
        timeout_s=STEP_ID_POLL_TIMEOUT_S,
        poll_s=STEP_ID_POLL_INTERVAL_S,
    )


def _create_step_id_fifo(extras: dict | None, step_name: str | None) -> Path | None:
    """Create a FIFO under the swarm jobs directory for step-id reporting."""
    if not extras or not step_name:
        return None
    swarm_dir = extras.get("swarm_directory")
    if not swarm_dir:
        return None
    jobs_dir = Path(swarm_dir) / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    fifo_path = jobs_dir / f"{step_name}.fifo"
    try:
        if fifo_path.exists():
            fifo_path.unlink()
        os.mkfifo(fifo_path, 0o600)
    except Exception:
        return None
    return fifo_path


def _read_step_id_from_fifo(path: Path, *, timeout_s: float) -> str | None:
    """Read a job.step id from a FIFO path."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except Exception:
        return None
    try:
        ready, _, _ = select.select([fd], [], [], max(timeout_s, 0.0))
        if not ready:
            return None
        data = os.read(fd, 256)
    finally:
        os.close(fd)
    if not data:
        return None
    match = re.search(rb"\b(\d+\.\d+)\b", data)
    if not match:
        return None
    return match.group(1).decode()


def _read_step_id_from_stream(stdout, *, timeout_s: float) -> str | None:
    """Best-effort read of a job.step id from a process stdout stream.

    Args:
        stdout: Process stdout stream.
        timeout_s: Maximum time to wait for a line.

    Returns:
        The parsed job.step id, or None if unavailable.
    """
    if stdout is None:
        return None
    try:
        fileno = stdout.fileno()
    except Exception:
        return None
    ready, _, _ = select.select([fileno], [], [], max(timeout_s, 0.0))
    if not ready:
        return None
    line = stdout.readline()
    if not line:
        return None
    match = re.search(r"\b(\d+\.\d+)\b", line)
    if not match:
        return None
    return match.group(1)


def _try_cleanup_fifo(path: Path) -> None:
    with contextlib.suppress(Exception):
        path.unlink(missing_ok=True)


def _build_step_name(base_name: str) -> str:
    """Build a short, unique Slurm step name.

    Args:
        base_name: Base job name.

    Returns:
        A Slurm-compatible step name.
    """
    suffix = str(int(time.time() * 1000))[-6:]
    name = f"{base_name}-step-{suffix}"
    return name[:128]


def _wait_for_step_id(
    *,
    job_id: int,
    step_name: str,
    timeout_s: float,
    poll_s: float,
) -> str | None:
    """Poll Slurm until a step id with the given name appears.

    Args:
        job_id: Parent Slurm job id (allocation).
        step_name: Step name passed to `srun --job-name`.
        timeout_s: Maximum time to wait.
        poll_s: Poll interval.

    Returns:
        The step id string (e.g., "12345.0") if found, else None.
    """
    deadline = time.time() + max(timeout_s, 0.0)
    while True:
        step_id = _query_step_id_squeue(job_id, step_name)
        if step_id:
            return step_id
        step_id = _query_step_id_sacct(job_id, step_name)
        if step_id:
            return step_id
        if time.time() >= deadline:
            return None
        time.sleep(poll_s)


def _query_step_id_squeue(job_id: int, step_name: str) -> str | None:
    """Query `squeue` for a step id by name.

    Args:
        job_id: Parent Slurm job id (allocation).
        step_name: Step name passed to `srun --job-name`.

    Returns:
        The step id string if found, else None.
    """
    res = subprocess.run(
        ["squeue", "-j", str(job_id), "--steps", "-h", "-o", "%i", "-n", step_name],
        check=False,
        text=True,
        capture_output=True,
    )
    if res.returncode != 0 or not res.stdout:
        return None
    return res.stdout.strip().splitlines()[0].strip() or None


def _query_step_id_sacct(job_id: int, step_name: str) -> str | None:
    """Query `sacct` for a step id by name.

    Args:
        job_id: Parent Slurm job id (allocation).
        step_name: Step name passed to `srun --job-name`.

    Returns:
        The step id string if found, else None.
    """
    res = subprocess.run(
        ["sacct", "-j", str(job_id), "--steps", "-o", "JobID,JobName", "-n", "-P"],
        check=False,
        text=True,
        capture_output=True,
    )
    if res.returncode != 0 or not res.stdout:
        return None
    for line in res.stdout.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split("|", 1)]
        if len(parts) != 2:
            continue
        job_step_id, job_name = parts
        if job_name == step_name and job_step_id:
            return job_step_id
    return None


def _slurm_query_state(external_id: str) -> str:
    """Query Slurm for the current state of a job or step."""

    def _run(cmd: list[str]) -> str:
        res = subprocess.run(cmd, check=False, text=True, capture_output=True)
        if res.returncode == 0 and res.stdout:
            return res.stdout.strip()
        return ""

    job_id, step_id = _split_step_id(external_id)

    # 1) Live state via squeue (prefer steps if we have one)
    if step_id is not None:
        for cmd in (
            ["squeue", "-j", f"{job_id}.{step_id}", "-h", "-o", "%T"],
            ["squeue", "-j", str(job_id), "--steps", "-h", "-o", "%T"],
            ["squeue", "-j", str(job_id), "-s", "-h", "-o", "%T"],
        ):
            out = _run(cmd)
            if out:
                state = _clean_slurm_state(out)
                if state and state != "STATE":
                    return state
    else:
        out = _run(["squeue", "-j", str(job_id), "-h", "-o", "%T"])
        if out:
            state = _clean_slurm_state(out)
            if state and state != "STATE":
                return state

    # 2) Terminal state via sacct
    sacct_cmd = ["sacct", "-j", external_id, "-o", "State", "-n", "-P"]
    if step_id is None:
        sacct_cmd.append("-X")
    out = _run(sacct_cmd)
    if out:
        for line in out.splitlines():
            state = _clean_slurm_state(line)
            if state and state != "STATE":
                return state

    return "UNKNOWN"


def _clean_slurm_state(raw: str) -> str:
    """Normalize a Slurm state string from command output."""
    token = raw.strip().split()[0] if raw.strip() else ""
    if not token:
        return ""
    token = token.split("+")[0]
    return token.upper()


def _split_step_id(external_id: str) -> tuple[str, str | None]:
    """Split a Slurm external ID into job ID and optional step ID."""
    if "." in external_id:
        job_id, step_id = external_id.split(".", 1)
        return job_id, step_id
    return external_id, None


def _normalize_slurm_state(state: str) -> JobStatus:
    """Map a Slurm state string to a normalized `JobStatus`."""
    if not state or state == "UNKNOWN":
        return JobStatus.PENDING

    pending = {"PENDING", "CONFIGURING", "REQUEUED", "RESIZING"}
    running = {"RUNNING", "COMPLETING", "SUSPENDED"}
    succeeded = {"COMPLETED"}
    cancelled = {"CANCELLED", "PREEMPTED", "STOPPED"}
    failed = {"FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "BOOT_FAIL"}

    if state in pending:
        return JobStatus.PENDING
    if state in running:
        return JobStatus.RUNNING
    if state in succeeded:
        return JobStatus.SUCCEEDED
    if state in cancelled:
        return JobStatus.CANCELLED
    if state in failed:
        return JobStatus.FAILED

    return JobStatus.FAILED


def _wait_for_slurm(
    external_id: str,
    *,
    timeout: float | None,
    poll_s: float,
) -> JobStatus:
    """Poll Slurm until a job/step reaches a terminal state or timeout."""
    start = time.time()
    last_status = JobStatus.PENDING
    while True:
        state = _slurm_query_state(external_id)
        status = _normalize_slurm_state(state)
        last_status = status
        if status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}:
            return status
        if timeout is not None and (time.time() - start) >= timeout:
            return last_status
        time.sleep(poll_s)


def _cancel_slurm(external_id: str) -> None:
    """Cancel a Slurm job or step by identifier."""
    subprocess.run(["scancel", str(external_id)], check=False)
