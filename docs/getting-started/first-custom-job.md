# Your first custom job

A job is a subclass of `SwarmJob` that implements one method:
`transform_items`. Batching, bounded concurrency, retries and checkpointing are
provided by the framework — see
[The SwarmJob lifecycle](../concepts/swarmjob-lifecycle.md).

:::{warning}
The Python API is still evolving; expect breaking changes before the stable
release.

Legacy `transform(df)`-based jobs are no longer supported. Implement
`transform_items(items)`, or rely on the `transform_streaming` that `SwarmJob`
provides.
:::

## Define the job

A job declares its configuration as a class-level `config` — an instance of
`JobConfig`, or of a subclass adding fields of its own via `config_class` —
instead of writing a constructor. Every field on `config` is readable and
writable as an attribute of the job, and any of them can be overridden when
the class is instantiated (`MyCustomSwarmJob(model="gpt-4", max_concurrency=8)`).

```python
import random
from typing import Any

from domyn_swarm import JobConfig, SwarmJob


class MyCustomSwarmJob(SwarmJob):
    """
    Example custom job.

    - Reads prompts from the `input_column_name` column (default: "messages")
    - Produces three output columns: completion, score, current_model
    - No checkpointing/I-O logic here: the runner handles that.
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
                    random.random(),                   # demo score
                    f"{self.model}_{temperature}",     # demo tag
                )
            )

        return results
```

The important parts:

- `config` declares the job's configuration. Every field on it — including
  `output_cols` — is readable and writable as an attribute of the job, and
  overridable at construction time. A job that needs fields of its own pairs a
  `JobConfig` subclass with a matching `config_class`; see the
  [SwarmJob API reference](../reference/api/jobs.md).
- `output_cols` declares the columns the job writes. Set it to a list when a job
  returns several values per row, as above.
- `transform_items` receives a list of items and must return results in the same
  order and of the same length.
- `self.client` is an `AsyncOpenAI` already pointed at the swarm endpoint, and
  `self.kwargs` carries the configured `request_params` — the provider
  parameters forwarded on every request.
- No checkpointing or I/O logic belongs here. The runner handles it.

## Run it from the CLI

Address the class as `<module>:<ClassName>`:

```shell
PYTHONPATH=. domyn-swarm job submit examples.scripts.custom_job:MyCustomJob \
   --config examples/configs/deepseek_r1_distill.yaml \
   --input examples/data/completion.parquet \
   --output results/output.parquet \
   --job-kwargs '{"request_params": {"temperature": 0.2}}'
```

`PYTHONPATH=.` is what makes a job class in the working directory importable by
the driver process.

## Run it from a script

To control the swarm's lifetime yourself, use `DomynLLMSwarm` as a context
manager:

```python
from pathlib import Path
from domyn_swarm import DomynLLMSwarm, DomynLLMSwarmConfig
from mypkg.jobs import MyCustomSwarmJob

cfg = DomynLLMSwarmConfig.read("config.yaml")

with DomynLLMSwarm(cfg=cfg) as swarm:
    job = MyCustomSwarmJob(
        endpoint=swarm.endpoint,
        model=swarm.model,
        max_concurrency=16,
        request_params={"temperature": 0.2},
    )
    swarm.submit_job(job, input_path=Path("prompts.parquet"), output_path=Path("answers.parquet"))
```

The endpoint is torn down when the block exits. Pass `delete_on_exit=False` to
keep the allocation alive and reattach later with
`DomynLLMSwarm.from_state(name)`.

## Next steps

- [SwarmJob API reference](../reference/api/jobs.md)
- [The SwarmJob lifecycle](../concepts/swarmjob-lifecycle.md) — what happens around your method
- [Checkpointing and resuming](../guides/checkpointing.md) — surviving a failed run
