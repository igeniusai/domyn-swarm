# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""`JobRunSpec` is the only way to pass run parameters to `submit_job`."""

import inspect
from pathlib import Path

from domyn_swarm.core.job_run import JobRunSpec
from domyn_swarm.core.swarm import DomynLLMSwarm


def test_submit_job_takes_run_spec_only() -> None:
    """No flat run parameters remain to drift out of sync with `JobRunSpec`."""
    params = inspect.signature(DomynLLMSwarm.submit_job).parameters
    assert set(params) == {"self", "job", "run"}
    assert params["run"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["run"].default is inspect.Parameter.empty


def test_only_the_paths_are_required() -> None:
    """A spec is constructible from the two paths alone."""
    spec = JobRunSpec(input_path=Path("in.parquet"), output_path=Path("out.parquet"))
    assert spec.num_shards == 1
    assert spec.runner == "pandas"
    assert spec.shard_mode == "id"
    assert spec.engine is None
    assert spec.job_resources is None
    assert spec.num_threads is None
    assert spec.checkpoint_interval is None


def test_cli_helpers_reexports_the_same_class() -> None:
    """The CLI imports the core class rather than defining its own."""
    from domyn_swarm.cli.job_helpers import JobRunSpec as CliJobRunSpec

    assert CliJobRunSpec is JobRunSpec
