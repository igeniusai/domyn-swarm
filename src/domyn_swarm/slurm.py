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
