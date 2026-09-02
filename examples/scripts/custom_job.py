# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""
This script is an example on how you can use the domyn-swarm API to
implement your own custom job, which is totally free in terms of implementation.

A custom job needs no constructor at all. Declare its configuration as a
class-level `config` (an instance of `JobConfig`, or of a subclass that adds
your own fields via `config_class`) and implement `transform_items`:

async def transform_items(items: list[Any]) -> list[Any]:
    pass

you can either run this job using the CLI

PYTHONPATH=. domyn-swarm job submit examples.scripts.custom_job:MyCustomSwarmJob \
   --config examples/configs/deepseek_r1_distill.yaml \
   --input examples/data/completion.parquet \
   --output results/output.parquet \
   --job-kwargs '{"request_params": {"temperature": 0.2}}'

or running a main module importing and instantiating this class.

Note: make sure that the package containing the import is on the path

e.g:

PYTHONPATH=. python examples/scripts/custom_main.py
"""

import random
from typing import Any

from domyn_swarm import JobConfig, SwarmJob


class MyCustomSwarmJob(SwarmJob):
    """
    Example custom job using the new SwarmJob API.

    - Reads prompts from the `input_column_name` column (default: "messages")
    - Produces three output columns: completion, score, current_model
    - No checkpointing/I-O logic here: the runner handles that.

    Every field set below already exists on `JobConfig`, so this job only
    overrides defaults and needs no `config_class`. A job that needs *new*
    fields declares its own `JobConfig` subclass and points `config_class` at
    it; see `SwarmJob`'s docstring.
    """

    config = JobConfig(
        input_column_name="messages",
        output_cols=["completion", "score", "current_model"],
        checkpoint_interval=16,
        max_concurrency=2,
        retries=5,
        timeout=600,
    )

    async def transform_items(self, items: list[Any]) -> list[tuple[str, float, str]]:
        """
        Pure transform: items -> results (same order, same length).
        Each item here is expected to be a prompt string.

        Returns:
            List of tuples: (completion_text, random_score, model_tag)
        """
        # Provider request parameters (e.g. temperature) are configured via
        # `request_params` and read back through `self.kwargs`.
        temperature = float(self.kwargs.get("temperature", 0.7))

        results: list[tuple[str, float, str]] = []

        # Note: The executor calls this for single items via `_call_unit`,
        # but we support lists to keep the contract general.
        for prompt in items:
            # Async OpenAI client already configured to hit the swarm endpoint
            resp = await self.client.completions.create(
                model=self.model,
                prompt=prompt,
                **self.kwargs,  # forward any extra OpenAI parameters
            )
            completion_text = resp.choices[0].text or ""
            results.append(
                (
                    completion_text,
                    random.random(),  # demo score
                    f"{self.model}_{temperature}",  # demo tag
                )
            )

        return results
