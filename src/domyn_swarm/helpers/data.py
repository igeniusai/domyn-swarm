import math
import time

from openai.types.chat.chat_completion import Choice
from typing import List, Tuple
from domyn_swarm import utils


import hashlib
import mmap
import os


def parquet_hash(
    path: str | utils.EnvPath,
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
    path: utils.EnvPath = utils.EnvPath(path)
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


def compute_hash(s: str, algorithm="sha256"):
    """
    Compute the hexadecimal hash digest of string `s` using `algorithm`.
    """
    h = hashlib.new(algorithm)
    h.update(s.encode("utf-8"))
    return h.hexdigest()


def generate_swarm_name() -> str:
    import random
    import string

    return f"""domyn-swarm-{int(time.time())}-{
        "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    }"""
