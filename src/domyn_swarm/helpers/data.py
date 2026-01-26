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

from _hashlib import HASH
import glob
import hashlib
import math
import mmap
import os
from pathlib import Path
import sys

from openai.types.chat.chat_completion import Choice

from domyn_swarm import utils
from domyn_swarm.helpers.patterns import expand_brace_ranges


def _hash_file(
    file_path: utils.EnvPath,
    hasher: HASH,
    *,
    block_size: int,
) -> HASH:
    """Update a hasher with a file's bytes.

    Args:
        file_path: File path to hash.
        hasher: Hashlib hasher instance.
        block_size: Chunk size used for 32-bit mmap fallbacks.

    Returns:
        Updated hasher.
    """
    file_size = file_path.stat().st_size
    with file_path.open("rb", buffering=0) as f:
        # On 64-bit Pythons (or “small” files) we can map the whole thing.
        if sys.maxsize > 2**32 or file_size < 2**31:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                hasher.update(mm)  # zero-copy→kernel pages it in lazily
        else:
            # 32-bit fall-back: slide a window across the file.
            offset = 0
            while offset < file_size:
                length = min(block_size, file_size - offset)
                with mmap.mmap(f.fileno(), length, offset=offset, access=mmap.ACCESS_READ) as mm:
                    hasher.update(mm)
                offset += length
    return hasher


def _expand_input_files(path_str: str) -> tuple[list[utils.EnvPath], bool]:
    """Expand a path string into a list of input files.

    Supports:
    - Brace-range expansion: `{0001..0100}`
    - Wildcard globs: `*`, `?`, `[...]`
    - Directory inputs: expands to all files under the directory

    Args:
        path_str: Input string.

    Returns:
        Tuple of (`files`, `pattern_mode`).
        `pattern_mode` is True when brace/glob expansion was used.
    """
    patterns = expand_brace_ranges(path_str)
    expanded = patterns != [path_str]
    has_wildcard = any(any(ch in pat for ch in ("*", "?", "[")) for pat in patterns)
    pattern_mode = expanded or has_wildcard

    if not pattern_mode:
        p = utils.EnvPath(path_str)
        if p.is_dir():
            return sorted(fp for fp in p.rglob("*") if fp.is_file()), True
        return [p], False

    matched: list[str] = []
    for pat in patterns:
        if any(ch in pat for ch in ("*", "?", "[")):
            matched.extend(glob.glob(pat))
        else:
            matched.append(pat)

    files: list[utils.EnvPath] = []
    for m in matched:
        p = utils.EnvPath(m)
        if p.is_dir():
            files.extend(sorted(fp for fp in p.rglob("*") if fp.is_file()))
        elif p.is_file():
            files.append(p)

    # De-dupe deterministically.
    unique_by_str = {str(p): p for p in files}
    return list(unique_by_str.values()), True


def _hash_file_set(files: list[utils.EnvPath], hasher: HASH, *, block_size: int) -> HASH:
    """Hash a set of files deterministically.

    The hash includes each file's relative path (relative to the common base directory) plus
    file contents. This makes the digest stable across different filesystem iteration orders.

    Args:
        files: File paths to hash.
        hasher: Hashlib hasher instance.
        block_size: Chunk size used for 32-bit mmap fallbacks.

    Returns:
        Updated hasher.
    """
    common = os.path.commonpath([str(p) for p in files])
    base_dir = utils.EnvPath(common)
    if base_dir.is_file():
        base_dir = utils.EnvPath(base_dir.parent)

    def _rel_key(p: utils.EnvPath) -> str:
        try:
            return p.relative_to(base_dir).as_posix()
        except Exception:
            return str(p)

    for p in sorted(files, key=_rel_key):
        rel = _rel_key(p)
        hasher.update(rel.encode("utf-8"))
        hasher = _hash_file(utils.EnvPath(p), hasher, block_size=block_size)
    return hasher


def parquet_hash(
    path: Path | str,
    algorithm: str = "blake2b",
    *,
    block_size: int = 128 << 20,  # 128 MiB windows if we have to chunk
) -> str:
    """Return a cryptographic hash of Parquet inputs (file, directory, or patterns).

    Parameters
    ----------
    path : Path | str
        Location of the Parquet file, a directory (parquet dataset), or a pattern.
        Supported patterns:
        - Bash-like numeric brace ranges: `file-{0001..0100}.parquet`
        - Wildcard globs (pandas-style): `data-*.parquet`
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
    env_path: utils.EnvPath = utils.EnvPath(path)
    h = hashlib.new(algorithm)
    if env_path.is_dir():
        files = sorted(utils.EnvPath(p) for p in env_path.rglob("*") if p.is_file())
        h = _hash_file_set(files, h, block_size=block_size)
        return h.hexdigest()[:8]

    files, pattern_mode = _expand_input_files(str(env_path))
    if pattern_mode:
        if not files:
            raise ValueError(f"No files matched pattern: {path}")
        if len(files) == 1:
            h = _hash_file(files[0], h, block_size=block_size)
            return h.hexdigest()[:8]
        h = _hash_file_set(files, h, block_size=block_size)
        return h.hexdigest()[:8]

    # Single concrete file path: preserve historic behavior (content-only).
    h = _hash_file(files[0], h, block_size=block_size)
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


def extract_token_logprobs(choice: Choice) -> list[float]:
    """
    Given a Choice with logprobs.content, pull out all the non-None logprobs.
    """
    if not (choice.logprobs and choice.logprobs.content):
        return []
    return [tl.logprob for tl in choice.logprobs.content if tl.logprob is not None]


def compute_perplexity_metrics(
    token_logprobs: list[float], bottom_k: int = 50
) -> tuple[float, float]:
    """
    Returns (perplexity, bottom_k_perplexity).
    """
    perp = compute_perplexity(token_logprobs)
    bottom_perp = compute_perplexity(sorted(token_logprobs)[:bottom_k])
    return perp, bottom_perp


def compute_hash(s: str, algorithm="sha256"):
    """
    Compute the hexadecimal hash digest of string `s` using `algorithm`.
    """
    h = hashlib.new(algorithm)
    h.update(s.encode("utf-8"))
    return h.hexdigest()


def get_device_slices(gpus_per_node: int, gpus_per_replica: int) -> list[str]:
    """
    Generate a list of device slices for the given number of GPUs per node
    and GPUs per replica.
    Each slice is a comma-separated string of GPU indices.
    """
    slices = []
    for i in range(0, gpus_per_node, gpus_per_replica):
        dev_ids = list(range(i, i + gpus_per_replica))
        if dev_ids[-1] >= gpus_per_node:
            dev_ids = [x for x in dev_ids if x < gpus_per_node]
            if dev_ids:
                # If we have a partial slice, we still want to add it
                # but we don't want to break the loop
                slices.append(",".join(str(x) for x in dev_ids))
            break

        slices.append(",".join(str(x) for x in dev_ids))
    return slices
