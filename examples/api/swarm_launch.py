

import os
from time import sleep
from domyn_swarm import DomynLLMSwarmConfig
from domyn_swarm.core.swarm_pool import create_swarm_pool
from domyn_swarm.jobs import ChatCompletionJob


def wait_for_children(pids):
    """Blocking wait for every PID in `pids` that is a child of this process."""
    for pid in pids:
        try:
            # The second argument of 0 means "block until it exits".
            _, status = os.waitpid(pid, 0)
            exit_code = os.WEXITSTATUS(status)
            print(f"PID {pid} finished with exit code {exit_code}")
        except ChildProcessError:
            # The PID is not a direct child of this process
            print(f"PID {pid} is not a child of this process; skipping.")
        except OSError as err:
            # Handles cases where the PID no longer exists, permissions issues, etc.
            print(f"Could not wait for PID {pid}: {err}")

# For simplicity, we use the same configuration for both swarms.
# In practice, you can use different configurations for each swarm.
# This example assumes you have a valid configuration file at the specified path.
configs = [
    DomynLLMSwarmConfig.read("examples/configs/deepseek_r1_distill.yaml"),
    DomynLLMSwarmConfig.read("examples/configs/deepseek_r1_distill.yaml")
]
pids = []
with create_swarm_pool(*configs) as swarms:
    for index, swarm in enumerate(swarms):
        # We create a job instance for each swarm.
        # Of course, you can customize the job parameters as needed.
        job = ChatCompletionJob(
            endpoint=swarm.endpoint,
            model=swarm.model,
        )
        # We submit the job to the swarm, specifying input and output paths.
        # Specifically, we use detach=True to submit the job immediately
        # and not block the main thread, allowing us to submit multiple jobs in parallel.
        # The job will run in the background, and we can collect the PIDs to wait
        # for their completion later.
        pid = swarm.submit_job(
            job, 
            input_path="examples/data/chat_completion.parquet",
            output_path=f"results_{index}.parquet",
            num_threads=2,
            detach=True
        )
        pids.append(pid)

# Now we wait for all submitted jobs to finish.
# This will block until all child processes (jobs) have completed.
wait_for_children(pids)