"""Custom exceptions."""


class DomynSwarmError(Exception):
    """Base class for all custom exceptions.

    Useful to catch all of them.
    """


class JobNotFoundError(DomynSwarmError):
    """Job ID not found in the DB."""

    def __init__(self, deployment_name: str):
        """Raise the JobNotFoundError.

        Args:
            deployment_name (str): Name of the job not found in the DB.
        """
        msg = f"Job '{deployment_name}' not found."
        super().__init__(msg)
