import pathlib
import socket
import subprocess
import sys
import tempfile
import time
from typing import List, Tuple
from rich import print as rprint
import jinja2
import hashlib
import mmap
import os
import math
from openai.types.chat.chat_completion import Choice
import logging


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
    cache_dir = pathlib.Path(os.getenv("TMPDIR", "/tmp")) / "cache"
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
    nginx_template: pathlib.Path,
    image_path: pathlib.Path,
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

    with tempfile.TemporaryDirectory(delete=False) as temp_dir:
        conf_path = pathlib.Path(temp_dir) / "nginx.conf"
        with open(conf_path, "w") as f:
            f.write(nginx_conf)
        html_path = pathlib.Path(temp_dir) / "html" / "index.html"
        html_path.parent.mkdir(parents=True, exist_ok=True)
        with open(html_path, "w") as f:
            f.write(
                f"<h1>Reverse proxy for {lb_node}</h1>\n<p>vLLM port: {vllm_port}</p>\n<p>Ray dashboard port: {ray_dashboard_port}</p>"
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


def parquet_hash(
    path: str | pathlib.Path,
    algorithm: str = "blake2b",
    *,
    block_size: int = 128 << 20,  # 128 MiB windows if we have to chunk
) -> str:
    """Return a cryptographic hash of an on-disk Parquet file.

    Parameters
    ----------
    path : str | Path
        Location of the Parquet file.
    algorithm : str, default "blake2b"
        Any algo accepted by ``hashlib.new`` (e.g. "blake2b", "sha256", "md5").
        "blake2b" is built-in, very fast, and 64-bit wide; if you install the
        third-party `blake3` package you can pass ``algorithm="blake3"`` for
        even higher multi-core throughput.
    block_size : int, default 128 MiB
        Size of each memory-mapped window when we *must* chunk (mainly for
        32-bit Python).  Use a multiple of the OS page size for best results.

    Returns
    -------
    str
        Hexadecimal digest of the file contents.
    """
    path: pathlib.Path = pathlib.Path(path)
    h = hashlib.new(algorithm)

    file_size = path.stat().st_size
    with path.open("rb", buffering=0) as f:
        # On 64-bit Pythons (or “small” files) we can map the whole thing.
        if os.sys.maxsize > 2**32 or file_size < 2**31:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                h.update(mm)  # zero-copy→kernel pages it in lazily
        else:
            # 32-bit fall-back: slide a window across the file.
            offset = 0
            while offset < file_size:
                length = min(block_size, file_size - offset)
                with mmap.mmap(
                    f.fileno(), length, offset=offset, access=mmap.ACCESS_READ
                ) as mm:
                    h.update(mm)
                offset += length

    return h.hexdigest()[:8]


def compute_perplexity(logprobs: list[float]) -> float:
    """
    Given the list of logprobs from the output of the model
    compute the perplexity score
    """
    if not logprobs:
        return float("inf")  # Avoid div by zero
    avg_neg_logprob = -sum(logprobs) / len(logprobs)
    return math.exp(avg_neg_logprob)


def extract_token_logprobs(choice: Choice) -> List[float]:
    """
    Given a Choice with logprobs.content, pull out all the non-None logprobs.
    """
    if not (choice.logprobs and choice.logprobs.content):
        return []
    return [tl.logprob for tl in choice.logprobs.content if tl.logprob is not None]


def compute_perplexity_metrics(
    token_logprobs: List[float], bottom_k: int = 50
) -> Tuple[float, float]:
    """
    Returns (perplexity, bottom_k_perplexity).
    """
    perp = compute_perplexity(token_logprobs)
    bottom_perp = compute_perplexity(sorted(token_logprobs)[:bottom_k])
    return perp, bottom_perp


def path_exists(path: str):
    return os.path.exists(path)


def is_folder(path: str):
    return pathlib.Path(path).is_dir()


def setup_logger(name: str = "app", level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Avoid duplicate logs

    if logger.handlers:
        return logger

    # Info and below → stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)
    stdout_handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
    )

    # Warnings and above → stderr
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
    )

    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)

    return logger
