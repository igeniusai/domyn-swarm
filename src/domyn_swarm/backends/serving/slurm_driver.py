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
import subprocess
import tempfile
from typing import Any

import jinja2

from domyn_swarm.helpers.data import get_device_slices
from domyn_swarm.helpers.io import is_folder, path_exists
from domyn_swarm.helpers.logger import setup_logger
import domyn_swarm.helpers.watchdog as watchdog_mod

logger = setup_logger(__name__)


class SlurmDriver:
    def __init__(self, cfg: Any):
        self.cfg = cfg

    def submit_replicas(
        self,
        job_name: str,
        replicas: int,
        nodes: int,
        gpus_per_node: int,
        gpus_per_replica: int,
        replicas_per_node: int,
        swarm_directory: str,
    ) -> int:
        """Submit the replica array job to Slurm.
        Returns the job ID of the submitted job."""
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.cfg.backend.template_path.parent),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        script_txt = env.get_template(self.cfg.backend.template_path.name).render(
            cfg=self.cfg,
            job_name=job_name,
            path_exists=path_exists,
            is_folder=is_folder,
            cuda_visible_devices=get_device_slices(gpus_per_node, gpus_per_replica),
            swarm_directory=swarm_directory,
            watchdog_script_path=Path(watchdog_mod.__file__).resolve().as_posix(),
        )

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sbatch") as fh:
            fh.write(script_txt)
            script_path = fh.name

        sbatch_cmd = ["sbatch", "--parsable", "--export=ALL"]
        array_spec = None
        if self.cfg.backend.requires_ray:
            if self.cfg.backend.requires_ray:
                logger.info(
                    "Detected gpus_per_replica > gpus_per_node with "
                    "multiple nodes: enabling Ray support."
                )
            array_spec = f"0-{replicas - 1}%{replicas}"
            # In this case, the nodes are the total number of nodes to be allocated
            # So we divide by replicas to get the number of nodes per array task
            sbatch_cmd.append(f"--nodes={nodes // replicas}")
        elif nodes and nodes >= 1 and replicas >= 1:
            array_spec = f"0-{nodes - 1}%{nodes}"
            sbatch_cmd.append("--nodes=1")
            sbatch_cmd.append(f"--ntasks-per-node={replicas_per_node}")

        if array_spec is not None:
            sbatch_cmd.extend(["--array", array_spec])
        sbatch_cmd.append(script_path)

        out = subprocess.check_output(sbatch_cmd, text=True).strip()
        job_id = out.split(";")[0]

        logger.info(f"Submitted replicas job {job_id} with command: {' '.join(sbatch_cmd)}")

        return int(job_id)

    def submit_endpoint(
        self, job_name: str, dep_jobid: int, replicas: int, swarm_directory: str
    ) -> int:
        """Submit the load balancer job to Slurm with a dependency on the replica array job."""
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.cfg.backend.template_path.parent),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        lb_script_txt = env.get_template("lb.sh.j2").render(
            cfg=self.cfg,
            job_name=job_name,
            dep_jobid=dep_jobid,
            replicas=replicas,
            swarm_directory=swarm_directory,
        )

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sbatch") as fh:
            fh.write(lb_script_txt)
            script_path = fh.name

        cmd = [
            "sbatch",
            "--parsable",
            "--dependency",
            f"after:{dep_jobid}",
            "--export",
            f"DEP_JOBID={dep_jobid}",
            script_path,
        ]
        out = subprocess.check_output(cmd, text=True).strip()
        logger.info(f"Submitted load balancer job {out} with command: {' '.join(cmd)}")
        return int(out)

    def get_node_from_jobid(self, jobid: int) -> str:
        """Get the head node of the job given its job ID."""
        nodespec = subprocess.check_output(
            ["squeue", "-j", str(jobid), "-h", "-O", "NodeList:2048"],
            text=True,
        ).strip()
        nodes = subprocess.check_output(
            ["scontrol", "show", "hostnames", nodespec],
            text=True,
        ).splitlines()

        if not nodes:
            raise RuntimeError(f"Could not find nodes for job ID {jobid}")
        head_node = nodes[0]
        logger.info(f"Job ID {jobid} is running on head node {head_node}")

        return head_node

    def get_job_state(self, jobid: int) -> str:
        """Get the current state of a Slurm job.

        This method attempts to determine the job state using multiple approaches
        in order of preference:

        1. Live states from `squeue` for active jobs
        2. Terminal states from `sacct` for completed jobs
        3. Fallback to `scontrol show job` parsing

        Args:
            jobid (int): The Slurm job ID to query.

        Returns:
            str: The job state as a string. Possible values include:
                - Live states: RUNNING, PENDING, CONFIGURING, COMPLETING, etc.
                - Terminal states: COMPLETED, FAILED, CANCELLED, TIMEOUT,
                  NODE_FAIL, OUT_OF_MEMORY, etc.
                - 'UNKNOWN' if the state cannot be determined.

        Note:
            The method tries multiple Slurm commands to handle cases where jobs
            may have transitioned between live and terminal states, or when
            different commands may be unavailable or return different information.
        """

        def _run(cmd: list[str]) -> str:
            try:
                res = subprocess.run(cmd, check=False, text=True, capture_output=True)
                if res.returncode == 0 and res.stdout:
                    return res.stdout.strip()
            except Exception:
                pass
            return ""

        # 1) Live state via squeue
        out = _run(["squeue", "-j", str(jobid), "-h", "-o", "%T"])
        if out:
            # %T returns a single token (e.g., RUNNING). Be cautious if someone changes the format.
            state = out.split()[0].upper()
            # Some sites prepend headers if -h is ignored; guard that:
            if state != "STATE":
                return state

        # 2) Final/terminal state via sacct (exclude steps with -X, machine-parsable with -P)
        #    This returns lines like "COMPLETED" or "FAILED" (one per job or step).
        #    With -X, we avoid step rows and keep the job record only.
        out = _run(["sacct", "-j", str(jobid), "-o", "State", "-n", "-X", "-P"])
        if out:
            # sacct may still return multiple lines in some schedulers (array parent/children).
            # Take the first non-empty token; with -P, fields are ';'-separated,
            # but we asked only State.
            for line in out.splitlines():
                s = line.strip()
                if not s:
                    continue
                # Drop trailing details like "CANCELLED by 12345" or "FAILED+"
                s = s.split()[0].split("+")[0].upper()
                if s and s != "STATE":
                    return s

        # 3) Fallback: scontrol show job -o (one-line). Parse JobState=...
        out = _run(["scontrol", "show", "job", "-o", str(jobid)])
        if out:
            # one line of key=val pairs, e.g., "... JobState=FAILED Reason=..."
            for field in out.split():
                if field.startswith("JobState="):
                    state = field.split("=", 1)[1].split()[0].upper()
                    if state:
                        return state

        return "UNKNOWN"
