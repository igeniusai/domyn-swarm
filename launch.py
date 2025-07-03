from pathlib import Path
from time import sleep
from domyn_swarm import DomynLLMSwarm, DomynLLMSwarmConfig
from domyn_swarm.jobs import ChatCompletionJob

config_1 = DomynLLMSwarmConfig.read(Path("examples/configs/deepseek_r1_distill.yaml"))
config_2 = DomynLLMSwarmConfig.read(Path("examples/configs/deepseek_r1_distill.yaml"))

for index, config in enumerate([config_1, config_2]):
    with DomynLLMSwarm(cfg=config) as swarm:
        job = ChatCompletionJob(
            endpoint=swarm.endpoint,
            model=swarm.model,
        )
        print(job.to_kwargs())
        swarm.submit_job(
            job,
            input_path=Path("examples/data/chat_completion.parquet"),
            output_path=Path(f"detached_results_{index}.parquet"),
            detach=True
        )

while True:
    sleep(60)
