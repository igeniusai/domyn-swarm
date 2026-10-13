# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Stand-in for the ``singularity`` commands that the Slurm templates run.

Every command runs natively in the calling container. Bind mounts become
symlinks at the destination path, so one compute node can host only one swarm
at a time. Options and subcommands that the templates do not use fail with exit
code 2, so a template change that needs new support is visible at once.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

BIND_FLAGS = frozenset({"--bind", "-B"})
IGNORED_FLAGS = frozenset({"--nv", "--writable-tmpfs"})
# A symlink over these would hide the node's own munge socket and Slurm files.
PROTECTED_DESTINATIONS = frozenset(
    {"/", "/run", "/tmp", "/sys", "/proc", "/dev", "/etc", "/usr", "/var", "/root", "/home"}
)


class ShimError(Exception):
    """An invocation that the shim does not support."""


def run_dir() -> Path:
    return Path(os.environ.get("FAKE_SINGULARITY_RUN_DIR", "/run/fake-singularity"))


def split_options(args: list[str]) -> tuple[list[str], list[str]]:
    binds: list[str] = []
    i = 0
    while i < len(args) and args[i].startswith("-"):
        flag = args[i]
        if flag in BIND_FLAGS:
            if i + 1 == len(args):
                raise ShimError(f"{flag} needs a value")
            binds.extend(spec for spec in args[i + 1].split(",") if spec)
            i += 2
        elif flag in IGNORED_FLAGS:
            i += 1
        else:
            raise ShimError(f"unsupported option: {flag}")
    return binds, args[i:]


def apply_bind(spec: str) -> None:
    src, _, rest = spec.partition(":")
    dst = rest.split(":", 1)[0] or src
    if dst == src:
        return
    if dst == "/" or dst.rstrip("/") in PROTECTED_DESTINATIONS:
        print(f"fake-singularity: skipping bind over {dst}", file=sys.stderr)
        return
    dst_path = Path(dst)
    if dst_path.is_dir() and not dst_path.is_symlink():
        shutil.rmtree(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    # Replicas on one node bind the same paths at the same moment; os.replace
    # swaps the link atomically where unlink-then-symlink would race.
    staging = dst_path.with_name(f".{dst_path.name}.{os.getpid()}")
    staging.symlink_to(src)
    os.replace(staging, dst_path)


def container_env() -> dict[str, str]:
    env = dict(os.environ)
    prefix = "SINGULARITYENV_"
    for key, value in os.environ.items():
        if key.startswith(prefix):
            env[key[len(prefix) :]] = value
    return env


def exec_command(args: list[str]) -> int:
    binds, rest = split_options(args)
    if len(rest) < 2:
        raise ShimError("exec needs an image and a command")
    for spec in binds:
        apply_bind(spec)
    command = rest[1:]
    os.execvpe(command[0], command, container_env())


def instance_start(args: list[str]) -> int:
    binds, rest = split_options(args)
    if len(rest) != 2:
        raise ShimError("instance start needs an image and a name")
    image, name = rest
    for spec in binds:
        apply_bind(spec)
    directory = run_dir()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{name}.log").open("ab") as log:
        process = subprocess.Popen(
            ["/bin/sh", image],
            env=container_env(),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (directory / f"{name}.pid").write_text(str(process.pid))
    return 0


def instance_stop(args: list[str]) -> int:
    if len(args) != 1:
        raise ShimError("instance stop needs a name")
    pid_file = run_dir() / f"{args[0]}.pid"
    if not pid_file.exists():
        print(f"fake-singularity: no instance named {args[0]}", file=sys.stderr)
        return 1
    with contextlib.suppress(ProcessLookupError):
        os.kill(int(pid_file.read_text()), signal.SIGTERM)
    pid_file.unlink()
    return 0


def main(argv: list[str]) -> int:
    try:
        match argv:
            case ["exec", *rest]:
                return exec_command(rest)
            case ["instance", "start", *rest]:
                return instance_start(rest)
            case ["instance", "stop", *rest]:
                return instance_stop(rest)
            case [command, *_]:
                raise ShimError(f"unsupported subcommand: {command}")
            case []:
                raise ShimError("missing subcommand")
    except ShimError as error:
        print(f"fake-singularity: {error}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
