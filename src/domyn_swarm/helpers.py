import pathlib
import socket
import subprocess
import sys
import tempfile
import time
from rich import print as rprint
import jinja2


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
        try:
            sock = socket.socket()
            sock.bind(("", port))
            sock.listen(1)
            sock.close()
            return port
        except OSError:
            continue
    raise IOError("No free ports available in range {}-{}".format(start, end))


def get_login_node_suffix() -> str:
    try:
        hostname = subprocess.check_output(["hostname"], text=True).strip()
        return hostname.split(".")[0][-2:]
    except subprocess.CalledProcessError as e:
        print(f"Error getting hostname: {e}", file=sys.stderr)
        sys.exit(1)


def run_command(command: str):
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
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
    nginx_template: pathlib.Path,
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
    sif_path: pathlib.Path, conf_path: pathlib.Path, html_path: pathlib.Path
):
    rprint(f"[INFO] Starting nginx from Singularity container: {sif_path}")
    logfile = conf_path.parent / pathlib.Path("nginx_singularity.log")
    with open(logfile, "w") as log:
        subprocess.Popen(
            ["singularity", "instance", "stop", f"nginx_instance"],
            stdout=log,
            stderr=log,
        )
        time.sleep(1)
        command = [
            "singularity",
            "instance",
            "start",
            "-B",
            "/etc/hosts:/etc/hosts",
            "-B",
            "/etc/resolv.conf:/etc/resolv.conf",
            "-B",
            f"{conf_path}:/etc/nginx/nginx.conf",
            "-B",
            f"{html_path}:/usr/share/nginx/html",
            str(sif_path),
            f"nginx_instance",
        ]
        rprint("Running:\n\t", " ".join(command))
        subprocess.Popen(command, stdout=log, stderr=log)

    rprint(f"[INFO] NGINX container started in background. Logs: {logfile}")


def launch_reverse_proxy(
    nginx_template: pathlib.Path,
    image_path: pathlib.Path,
    node: str,
    vllm_port: int,
    ray_dashboard_port: int,
):
    port = get_unused_port()
    rprint(f"[INFO] Launching reverse proxy on port {port}...")
    nginx_conf = generate_nginx_config(
        nginx_template,
        host=node,
        public_port=port,
        vllm_port=vllm_port,
        ray_port=ray_dashboard_port,
    )

    with tempfile.TemporaryDirectory(delete=False) as temp_dir:
        conf_path = pathlib.Path(temp_dir) / "nginx.conf"
        with open(conf_path, "w") as f:
            f.write(nginx_conf)
        html_path = pathlib.Path(temp_dir) / "index.html"
        with open(html_path, "w") as f:
            f.write(
                f"<h1>Reverse proxy for {node}</h1>\n<p>vLLM port: {vllm_port}</p>\n<p>Ray dashboard port: {ray_dashboard_port}</p>"
            )
        launch_nginx_singularity(
            sif_path=image_path,
            conf_path=conf_path,
            html_path=temp_dir,
        )
    user = run_command("whoami").strip()
    login_node_suffix = get_login_node_suffix()
    ssh_cmd = generate_ssh_tunnel_cmd(
        user=user,
        localhost_port=port,
        nginx_port=port,
        login_node_suffix=login_node_suffix,
    )

    rprint(
        "\n[INFO] Run the following command in your local terminal to create the SSH tunnel:"
    )
    rprint(ssh_cmd)
    rprint("[DONE]")
