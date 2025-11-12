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
import socket
import subprocess
import sys
import tempfile
import time

import jinja2
import requests
from requests.exceptions import RequestException
from rich import print as rprint

from domyn_swarm import utils
from domyn_swarm.helpers.logger import setup_logger

logger = setup_logger(__name__)


def get_unused_port(start=50000, end=65535):
    """
    Find an unused port in the specified range.
    Args:
        start (int): Start of the port range (inclusive).
        end (int): End of the port range (inclusive).
    Returns:
        int: An unused port number.
    Raises:
        IOError: If no free ports are available in the specified range.
    """
    for port in range(start, end + 1):
        sock = socket.socket()
        try:
            sock.bind(("", port))
            sock.listen(1)
            sock.close()
            return port
        except OSError:
            sock.close()
            continue
    raise OSError(f"No free ports available in range {start}-{end}")


def get_login_node_suffix() -> str:
    try:
        hostname = subprocess.check_output(["hostname"], text=True).strip()
        return hostname.split(".")[0][-2:]
    except subprocess.CalledProcessError as e:
        print(f"Error getting hostname: {e}", file=sys.stderr)
        sys.exit(1)


def run_command(command: str):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, errors = process.communicate()
    return_code = process.returncode
    assert return_code == 0, f"Command failed with error: {errors.decode('utf-8')}"
    return output.decode("utf-8").strip()


def generate_ssh_tunnel_cmd(
    user: str, localhost_port: int, nginx_port: int, login_node_suffix: str
) -> str:
    return (
        f"ssh -N -L {localhost_port}:login{login_node_suffix}.leonardo.local:{nginx_port} "
        f"{user}@login{login_node_suffix}-ext.leonardo.cineca.it"
    )


def generate_nginx_config(
    nginx_template: utils.EnvPath,
    host: str,
    public_port: int,
    vllm_port: int,
    ray_port: int,
) -> str:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(nginx_template.parent),
        autoescape=jinja2.select_autoescape(["html", "xml"]),
    )
    conf_str = env.get_template(nginx_template.name).render(
        port=public_port, host=host, vllm_port=vllm_port, ray_port=ray_port
    )

    return conf_str


def launch_nginx_singularity(
    sif_path: utils.EnvPath, conf_path: utils.EnvPath, html_path: utils.EnvPath
):
    rprint(f"[INFO] Starting nginx from Singularity container: {sif_path}")
    logfile = conf_path.parent / utils.EnvPath("nginx_singularity.log")
    cache_dir = utils.EnvPath(os.getenv("TMPDIR", "/tmp")) / "cache"
    os.makedirs(str(cache_dir), exist_ok=True)
    with open(logfile, "w") as log:
        subprocess.Popen(
            ["singularity", "instance", "stop", "nginx_instance"],
            stdout=log,
            stderr=log,
        )
        time.sleep(1)
        command = [
            "singularity",
            "instance",
            "start",
            "--writable-tmpfs",
            "-B",
            "/etc/hosts:/etc/hosts",
            "-B",
            "/etc/resolv.conf:/etc/resolv.conf",
            "-B",
            f"{conf_path}:/etc/nginx/nginx.conf",
            "-B",
            f"{html_path}:/usr/share/nginx/html",
            "-B",
            f"{cache_dir}:/var/cache/nginx",
            str(sif_path),
            "nginx_instance",
        ]
        rprint("Running:\n\t", " ".join(command))
        subprocess.Popen(command, stdout=log, stderr=log)

    rprint(f"[INFO] NGINX container started in background. Logs: {logfile}")


def launch_reverse_proxy(
    nginx_template: utils.EnvPath,
    image_path: utils.EnvPath,
    lb_node: str,
    head_node: str,
    vllm_port: int,
    ray_dashboard_port: int,
):
    port = get_unused_port()
    rprint(f"[INFO] Launching reverse proxy on port {port}...")
    nginx_conf = generate_nginx_config(
        nginx_template,
        host=head_node,
        public_port=0,
        vllm_port=vllm_port,
        ray_port=ray_dashboard_port,
    )

    # Deletion control is only available from Python 3.12
    if sys.version_info >= (3, 12):
        tmp_directory = tempfile.TemporaryDirectory(delete=False)
    else:
        tmp_directory = tempfile.TemporaryDirectory()

    with tmp_directory as temp_dir:
        conf_path = utils.EnvPath(temp_dir) / "nginx.conf"
        with open(conf_path, "w") as f:
            f.write(nginx_conf)
        html_path = utils.EnvPath(temp_dir) / "html" / "index.html"
        html_path.parent.mkdir(parents=True, exist_ok=True)
        with open(html_path, "w") as f:
            f.write(
                f"<h1>Reverse proxy for {lb_node}</h1>\n<p>vLLM port: "
                f"{vllm_port}</p>\n<p>Ray dashboard port: {ray_dashboard_port}</p>"
            )
        launch_nginx_singularity(
            sif_path=image_path,
            conf_path=conf_path,
            html_path=html_path,
        )
    user = run_command("whoami").strip()
    login_node_suffix = get_login_node_suffix()
    ssh_cmd = generate_ssh_tunnel_cmd(
        user=user,
        localhost_port=port,
        nginx_port=port,
        login_node_suffix=login_node_suffix,
    )

    rprint("\n[INFO] Run the following command in your local terminal to create the SSH tunnel:")
    rprint(ssh_cmd)
    rprint("[DONE]")


def is_endpoint_healthy(endpoint: str, timeout: float = 2.0) -> bool:
    """
    Check if an NGINX server is healthy by sending a GET request.

    Args:
        endpoint (str): The URL of the NGINX server (e.g., http://localhost:80).
        timeout (float): Timeout in seconds for the request.

    Returns:
        bool: True if the server responds with 2xx status code, False otherwise.
    """
    try:
        response = requests.get(endpoint, timeout=timeout)
        return response.status_code >= 200 and response.status_code < 300
    except RequestException:
        return False
