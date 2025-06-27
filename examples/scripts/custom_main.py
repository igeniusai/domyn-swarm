from pathlib import Path
from domyn_swarm import DomynLLMSwarm, DomynLLMSwarmConfig
from examples.scripts.custom_job import MyCustomSwarmJob
from rich import print as rprint

config_path = Path("examples/configs/deepseek_r1_distill.yaml")
input_path = Path("examples/data/completion.parquet")
output_path = Path("results/output.parquet")
config = DomynLLMSwarmConfig.read(config_path)

# This will allocate the resources and then submit the job
# We will also delete the cluster once the job has finished
# by setting delete_on_exit=True
with DomynLLMSwarm(cfg=config, delete_on_exit=True) as swarm:
    job = MyCustomSwarmJob(
        endpoint=swarm.endpoint,
        model=swarm.model,
        # 16 concurrent requests to the LLM
        parallel=16,
        # You can add custom keyword arguments, which you
        # can reference in you transform implementation by calling
        # self.kwargs
        temperature=0.2,
    )
    rprint(job.to_kwargs())

    swarm.submit_job(job, input_path=input_path, output_path=output_path)
