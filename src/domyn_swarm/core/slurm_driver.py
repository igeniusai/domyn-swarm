import os
import subprocess
import tempfile

import jinja2

from domyn_swarm.helpers.data import get_device_slices
from domyn_swarm.helpers.io import is_folder, path_exists
from domyn_swarm.helpers.logger import setup_logger
from domyn_swarm.models.swarm import DomynLLMSwarmConfig

logger = setup_logger(__name__)


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
            cuda_visible_devices=get_device_slices(
                self.cfg.gpus_per_node, self.cfg.gpus_per_replica
            ),
        )

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sbatch") as fh:
            fh.write(script_txt)
            script_path = fh.name

        os.makedirs(self.cfg.log_directory / job_name, exist_ok=True)
        sbatch_cmd = ["sbatch", "--parsable"]
        array_spec = None
        if self.cfg.requires_ray:
            array_spec = f"0-{self.cfg.replicas - 1}%{self.cfg.replicas}"
        elif self.cfg.nodes >= 1 and self.cfg.replicas > 1:
            array_spec = f"0-{self.cfg.nodes - 1}%{self.cfg.nodes}"
            sbatch_cmd.append("--nodes=1")
            sbatch_cmd.append(f"--ntasks-per-node={self.cfg.replicas_per_node}")

        if array_spec is not None:
            sbatch_cmd.extend(["--array", array_spec])
        sbatch_cmd.append(script_path)

        logger.info(f"Submitting job with command: {' '.join(sbatch_cmd)}")

        out = subprocess.check_output(sbatch_cmd, text=True).strip()
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
