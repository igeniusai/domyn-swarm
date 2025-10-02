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

import pandas as pd

from domyn_swarm.jobs import SwarmJob


class MyCustomSwarmJob(SwarmJob):
    def __init__(
        self,
        *,
        endpoint=None,
        model="",
        input_column_name="messages",
        output_column_name="result",
        checkpoint_interval=16,
        max_concurrency=2,
        retries=5,
        **extra_kwargs,
    ):
        super().__init__(
            endpoint=endpoint,
            model=model,
            input_column_name=input_column_name,
            output_column_name=output_column_name,
            checkpoint_interval=checkpoint_interval,
            max_concurrency=max_concurrency,
            retries=retries,
            **extra_kwargs,
        )
        self.output_column_name = ["completion", "score", "current_model"]

    async def transform(self, df: pd.DataFrame):
        """
        You can do whatever you want inside this function, as long as it returns a pd.DataFrame
        """

        async def _call(prompt: str) -> str:
            from openai.types.completion import Completion

            # Default client is pointing to the endpoint deployed by domyn-swarm and defined in the
            # domyn-swarm config
            resp: Completion = await self.client.completions.create(
                model=self.model, prompt=prompt, **self.kwargs
            )
            temperature = self.kwargs["temperature"]

            return resp.choices[0].text, random.random(), self.model + f"_{temperature}"

        # Call the endpoint batching the calls asynchronously in the background
        await self.batched(df["messages"].tolist(), _call)
