# Copyright 2025 iGenius S.p.A
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

from rich import print as rprint

from domyn_swarm import DomynLLMSwarm, DomynLLMSwarmConfig
from examples.scripts.custom_job import MyCustomSwarmJob

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
        max_concurrency=16,
        # You can add custom keyword arguments, which you
        # can reference in you transform implementation by calling
        # self.kwargs
        temperature=0.2,
    )
    rprint(job.to_kwargs())

    swarm.submit_job(job, input_path=input_path, output_path=output_path)
