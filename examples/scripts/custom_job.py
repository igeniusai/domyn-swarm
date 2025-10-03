"""
This script is an example on how you can use the domyn-swarm API to
implement your own custom job, which is totally free in terms of implementation.

Your custom job must implement an async transform method, with the signature

async def transform(df: pd.DataFrame):
    pass

you can either run this job using the CLI

PYTHONPATH=. domyn-swarm job submit examples.scripts.custom_job:MyCustomSwarmJob \
   --config examples/configs/deepseek_r1_distill.yaml \
   --input examples/data/completion.parquet \
   --output results/output.parquet \
   --job-kwargs '{"temperature": 0.2}'

or running a main module importing and instantiating this class.

Note: make sure that the package containing the import is on the path

e.g:

PYTHONPATH=. python examples/scripts/custom_main.py
"""

import random
from typing import Any

from domyn_swarm.jobs import SwarmJob


class MyCustomSwarmJob(SwarmJob):
    """
    Example custom job using the new SwarmJob API.

    - Reads prompts from the `input_column_name` column (default: "messages")
    - Produces three output columns: completion, score, current_model
    - No checkpointing/I-O logic here: the runner handles that.
    """

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        model: str = "",
        input_column_name: str = "messages",
        # We'll override outputs to three columns
        output_column_name: str | list[str] = "result",
        checkpoint_interval: int = 16,
        max_concurrency: int = 2,
        retries: int = 5,
        timeout: float = 600,
        **extra_kwargs: Any,
    ):
        # Initialize the base job (creates self.client, stores kwargs, etc.)
        super().__init__(
            endpoint=endpoint,
            model=model,
            input_column_name=input_column_name,
            output_column_name=output_column_name,
            checkpoint_interval=checkpoint_interval,
            max_concurrency=max_concurrency,
            retries=retries,
            timeout=timeout,
            **extra_kwargs,
        )
        # Our job returns 3 values per item
        self.output_column_name = ["completion", "score", "current_model"]

    async def transform_items(self, items: list[Any]) -> list[tuple[str, float, str]]:
        """
        Pure transform: items -> results (same order, same length).
        Each item here is expected to be a prompt string.

        Returns:
            List of tuples: (completion_text, random_score, model_tag)
        """
        # You can pass OpenAI params via job kwargs (e.g., temperature)
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
