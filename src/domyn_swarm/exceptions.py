# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Custom exceptions."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from domyn_swarm.config.preflight import PathProblem


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


class SwarmConfigPathError(DomynSwarmError):
    """One or more config fields point at paths that cannot be used."""

    def __init__(self, problems: "Sequence[PathProblem]"):
        """Raise the SwarmConfigPathError.

        Args:
            problems (Sequence[PathProblem]): The unusable paths found while
                checking the config, in config order.
        """
        from domyn_swarm.config.preflight import format_path_problems

        self.problems = list(problems)
        super().__init__(format_path_problems(self.problems))
