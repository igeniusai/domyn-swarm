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

from datetime import timedelta
import subprocess


def parse_slurm_time_limit(value: str) -> timedelta | None:
    """Parse a Slurm ``time_limit`` string into a ``timedelta``.

    Accepts the formats documented in ``sbatch(1)``:
    ``minutes``, ``minutes:seconds``, ``hours:minutes:seconds``,
    ``days-hours``, ``days-hours:minutes``, ``days-hours:minutes:seconds``.
    Returns ``None`` if the input cannot be parsed.
    """
    if not value or not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None

    days = 0
    if "-" in s:
        d_str, _, rest = s.partition("-")
        try:
            days = int(d_str)
        except ValueError:
            return None
        s = rest

    parts = s.split(":") if s else []
    try:
        nums = [int(p) for p in parts] if parts else []
    except ValueError:
        return None

    hours = minutes = seconds = 0
    if "-" in value:
        # days-hours[:minutes[:seconds]]
        if len(nums) == 1:
            hours = nums[0]
        elif len(nums) == 2:
            hours, minutes = nums
        elif len(nums) == 3:
            hours, minutes, seconds = nums
        else:
            return None
    else:
        # minutes | minutes:seconds | hours:minutes:seconds
        if len(nums) == 1:
            minutes = nums[0]
        elif len(nums) == 2:
            minutes, seconds = nums
        elif len(nums) == 3:
            hours, minutes, seconds = nums
        else:
            return None

    return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)


def get_job_status(job_id: int) -> str:
    """
    Get the status of a job in Slurm.

    Args:
        job_id (str): The ID of the job to check.

    Returns:
        str: The status of the job.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["squeue", "--job", str(job_id), "--noheader", "--format=%T"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to get job status for {job_id}: {e}") from e


def is_job_running(job_id: str):
    """
    Given a job id, check if the job is in running state
    (needed to retrieve hostname from logs)
    """
    command = "squeue --me --states=R | awk '{print $1}' | tail -n +2"
    my_running_jobs = subprocess.run(
        command, shell=True, text=True, capture_output=True
    ).stdout.splitlines()
    return job_id in my_running_jobs
