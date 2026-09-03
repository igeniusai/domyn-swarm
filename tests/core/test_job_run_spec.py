# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""`JobRunSpec` covers every run parameter `submit_job` accepts."""

import dataclasses
import inspect
from pathlib import Path

from domyn_swarm.core.job_run import JobRunSpec
from domyn_swarm.core.swarm import DomynLLMSwarm


def test_submit_job_takes_run_and_legacy_only() -> None:
    """`submit_job` takes `run=JobRunSpec(...)` and a `**legacy` catch-all.

    No flat run parameters remain to drift out of sync with `JobRunSpec`.
    """
    params = inspect.signature(DomynLLMSwarm.submit_job).parameters
    assert set(params) == {"self", "job", "run", "legacy"}
    assert params["legacy"].kind is inspect.Parameter.VAR_KEYWORD


def test_spec_covers_every_documented_legacy_keyword() -> None:
    """The flat keywords `submit_job`'s docstring lists are exactly the spec's
    fields, in both directions."""
    documented_legacy_names = {
        "input_path",
        "output_path",
        "num_shards",
        "shard_output",
        "detach",
        "limit",
        "mail_user",
        "checkpoint_dir",
        "checkpoint_interval",
        "no_resume",
        "no_checkpointing",
        "runner",
        "engine",
        "shard_mode",
        "global_resume",
        "job_resources",
        "checkpoint_tag",
        "ray_address",
        "num_threads",
    }
    fields = {f.name for f in dataclasses.fields(JobRunSpec)}
    assert documented_legacy_names == fields


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
