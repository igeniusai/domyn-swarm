"""
This script is an example on how you can use the domyn-swarm API to
implement your own custom job, which is totally free in terms of implementation.

Your custom job must implement an async transform method, with the signature

async def transform(df: pd.DataFrame) -> pd.DataFrame
    pass

you can either run this job using the CLI

PYTHONPATH=. domyn-swarm submit job examples.scripts.custom_job:MyCustomSwarmJob \
   --config examples/configs/deepseek_r1_distill.yaml \
   --input examples/data/completion.parquet \
   --output results/output.parquet \
   --job-kwargs '{"temperature": 0.2}'

or running a main module importing and instantiating this class.

Note: make sure that the package containing the import is on the path

e.g:

PYTHONPATH=. python examples/scripts/custom_main.py
"""

from domyn_swarm.jobs import SwarmJob
import pandas as pd
import random


class MyCustomSwarmJob(SwarmJob):
    def __init__(
        self,
        *,
        endpoint=None,
        model="",
        input_column_name="messages",
        output_column_name="result",
        batch_size=16,
        parallel=2,
        retries=5,
        **extra_kwargs,
    ):
        super().__init__(
            endpoint=endpoint,
            model=model,
            input_column_name=input_column_name,
            output_column_name=output_column_name,
            batch_size=batch_size,
            parallel=parallel,
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
        df["completion"] = await self.batched(df["messages"].tolist(), _call)

        return df
