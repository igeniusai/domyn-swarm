import os
import subprocess
import tempfile

import jinja2

from domyn_swarm.helpers.io import is_folder, path_exists
from domyn_swarm.models.swarm import DomynLLMSwarmConfig


class SlurmDriver:
    def __init__(self, cfg: DomynLLMSwarmConfig):
        self.cfg = cfg

    def submit_replicas(self, job_name: str) -> int:
        """Submit the replica array job to Slurm.
        Returns the job ID of the submitted job."""
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.cfg.template_path.parent),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        script_txt = env.get_template(self.cfg.template_path.name).render(
            cfg=self.cfg,
            job_name=job_name,
            path_exists=path_exists,
            is_folder=is_folder,
        )

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sbatch") as fh:
            fh.write(script_txt)
            script_path = fh.name

        os.makedirs(self.cfg.log_directory / job_name, exist_ok=True)
        array_spec = f"0-{self.cfg.replicas - 1}%{self.cfg.replicas}"
        out = subprocess.check_output(
            ["sbatch", "--parsable", "--array", array_spec, script_path], text=True
        ).strip()
        job_id = out.split(";")[0]

        os.makedirs(self.cfg.home_directory / "swarms" / job_id, exist_ok=True)
        return int(job_id)

    def submit_lb(self, job_name: str, dep_jobid: int) -> int:
        """Submit the load balancer job to Slurm with a dependency on the replica array job."""
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.cfg.template_path.parent),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        lb_script_txt = env.get_template("lb.sh.j2").render(
            cfg=self.cfg, job_name=job_name, dep_jobid=dep_jobid
        )

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sbatch") as fh:
            fh.write(lb_script_txt)
            script_path = fh.name

        out = subprocess.check_output(
            [
                "sbatch",
                "--parsable",
                "--dependency",
                f"after:{dep_jobid}",
                "--export",
                f"DEP_JOBID={dep_jobid}",
                script_path,
            ],
            text=True,
        ).strip()
        return int(out)

    def get_node_from_jobid(self, jobid: int) -> str:
        """Get the head node of the job given its job ID."""
        nodespec = subprocess.check_output(
            ["squeue", "-j", str(jobid), "-h", "-O", "NodeList:2048"],
            text=True,
        ).strip()
        head_node = subprocess.check_output(
            ["scontrol", "show", "hostnames", nodespec],
            text=True,
        ).splitlines()[0]
        return head_node
