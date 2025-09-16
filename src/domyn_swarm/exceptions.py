"""Custom exceptions."""


class DomynSwarmError(Exception):
    """Base class for all custom exceptions.

    Useful to catch all of them.
    """


class JobNotFoundError(DomynSwarmError):
    """Job ID not found in the DB."""

    def __init__(self, jobid: int):
        """Raise the JobNotFoundError.

        Args:
            jobid (int): ID of the job not found in the DB.
        """
        msg = f"Job '{jobid}' not found."
        super().__init__(msg)
