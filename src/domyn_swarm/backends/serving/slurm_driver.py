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

import os
import subprocess
import tempfile
from typing import Any

import jinja2

from domyn_swarm.helpers.data import get_device_slices
from domyn_swarm.helpers.io import is_folder, path_exists
from domyn_swarm.helpers.logger import setup_logger

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
        )

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sbatch") as fh:
            fh.write(script_txt)
            script_path = fh.name

        os.makedirs(self.cfg.backend.log_directory / job_name, exist_ok=True)
        sbatch_cmd = ["sbatch", "--parsable", "--export=ALL"]
        array_spec = None
        if self.cfg.backend.requires_ray:
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

        logger.info(
            f"Submitted replicas job {job_id} with command: {' '.join(sbatch_cmd)}"
        )

        os.makedirs(self.cfg.backend.home_directory / "swarms" / job_id, exist_ok=True)
        return int(job_id)

    def submit_endpoint(self, job_name: str, dep_jobid: int, replicas: int) -> int:
        """Submit the load balancer job to Slurm with a dependency on the replica array job."""
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.cfg.backend.template_path.parent),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        lb_script_txt = env.get_template("lb.sh.j2").render(
            cfg=self.cfg, job_name=job_name, dep_jobid=dep_jobid, replicas=replicas
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
        """Return Slurm job state (e.g., RUNNING, PENDING, FAILED, CANCELLED, TIMEOUT, UNKNOWN)."""
        try:
            out = subprocess.check_output(
                ["squeue", "-j", str(jobid), "-h", "-o", "%T"], text=True
            ).strip()
            if out:
                state = out.split()[0]
                if state == "State":
                    return "UNKNOWN"
                return state
            return "UNKNOWN"
        except Exception:
            return "UNKNOWN"
