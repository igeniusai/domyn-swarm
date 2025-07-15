import subprocess


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
    """Given job id, check if the job is in eunning state (needed to retrieve hostname from logs)"""
    command = "squeue --me --states=R | awk '{print $1}' | tail -n +2"
    my_running_jobs = subprocess.run(
        command, shell=True, text=True, capture_output=True
    ).stdout.splitlines()
    return job_id in my_running_jobs
