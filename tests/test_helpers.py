# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import math

import pandas as pd
import pytest

from domyn_swarm.helpers.data import (
    compute_perplexity,
    compute_perplexity_metrics,
    get_device_slices,
    parquet_hash,
)


def test_parquet_hash_blake2b(parquet_file):
    digest = parquet_hash(parquet_file, algorithm="blake2b")
    assert isinstance(digest, str)
    assert len(digest) == 8
    assert all(c in "0123456789abcdef" for c in digest)


def test_parquet_hash_sha256(parquet_file):
    digest = parquet_hash(parquet_file, algorithm="sha256")
    assert isinstance(digest, str)
    assert len(digest) == 8


def test_parquet_hash_md5(parquet_file):
    digest = parquet_hash(parquet_file, algorithm="md5")
    assert isinstance(digest, str)
    assert len(digest) == 8


def test_parquet_hash_invalid_algorithm(parquet_file):
    with pytest.raises(ValueError):
        parquet_hash(parquet_file, algorithm="notarealhash")


def test_parquet_hash_identical_files(tmp_path):
    df = pd.DataFrame({"x": [10, 20], "y": [0.1, 0.2]})
    f1 = tmp_path / "file1.parquet"
    f2 = tmp_path / "file2.parquet"
    df.to_parquet(f1)
    df.to_parquet(f2)

    hash1 = parquet_hash(f1)
    hash2 = parquet_hash(f2)
    assert hash1 == hash2


def test_parquet_hash_different_files(tmp_path):
    df1 = pd.DataFrame({"x": [1]})
    df2 = pd.DataFrame({"x": [2]})
    f1 = tmp_path / "a.parquet"
    f2 = tmp_path / "b.parquet"
    df1.to_parquet(f1)
    df2.to_parquet(f2)

    hash1 = parquet_hash(f1)
    hash2 = parquet_hash(f2)
    assert hash1 != hash2


def test_parquet_hash_brace_range_matches_directory_hash(tmp_path):
    df = pd.DataFrame({"x": [10, 20]})
    for i in range(1, 4):
        df.to_parquet(tmp_path / f"file-{i:04d}.parquet", index=False)

    dir_hash = parquet_hash(tmp_path)
    pattern_hash = parquet_hash(tmp_path / "file-{0001..0003}.parquet")
    assert pattern_hash == dir_hash


def test_parquet_hash_glob_matches_directory_hash(tmp_path):
    df = pd.DataFrame({"x": [10, 20]})
    for i in range(1, 4):
        df.to_parquet(tmp_path / f"file-{i:04d}.parquet", index=False)

    dir_hash = parquet_hash(tmp_path)
    glob_hash = parquet_hash(tmp_path / "file-*.parquet")
    assert glob_hash == dir_hash


def test_parquet_hash_pattern_no_match_raises(tmp_path):
    with pytest.raises(ValueError, match="No files matched pattern"):
        parquet_hash(tmp_path / "missing-{0001..0003}.parquet")


def test_perplexity_typical_case():
    logprobs = [-1.0, -2.0, -1.5]
    result = compute_perplexity(logprobs)
    expected = math.exp(-sum(logprobs) / len(logprobs))
    assert math.isclose(result, expected)


def test_perplexity_empty_list_returns_inf():
    result = compute_perplexity([])
    assert result == float("inf")


def test_perplexity_single_value():
    logprobs = [-2.0]
    result = compute_perplexity(logprobs)
    assert math.isclose(result, math.exp(2.0))


def test_perplexity_all_zero_logprobs():
    logprobs = [0.0, 0.0, 0.0]
    result = compute_perplexity(logprobs)
    assert math.isclose(result, 1.0)


def test_perplexity_metrics_normal_case():
    logprobs = [-1.0, -2.0, -3.0, -4.0, -5.0]
    perp, bottom_perp = compute_perplexity_metrics(logprobs, bottom_k=3)

    # Expected: perplexity over all, and perplexity over 3 smallest (most negative) logprobs
    expected_full = math.exp(-sum(logprobs) / len(logprobs))
    bottom = sorted(logprobs)[:3]
    expected_bottom = math.exp(-sum(bottom) / len(bottom))

    assert math.isclose(perp, expected_full)
    assert math.isclose(bottom_perp, expected_bottom)


def test_perplexity_metrics_empty_list():
    perp, bottom_perp = compute_perplexity_metrics([], bottom_k=5)
    assert perp == float("inf")
    assert bottom_perp == float("inf")


def test_perplexity_metrics_bottom_k_larger_than_list():
    logprobs = [-1.0, -1.5]
    perp, bottom_perp = compute_perplexity_metrics(logprobs, bottom_k=10)

    assert math.isclose(perp, math.exp(-sum(logprobs) / len(logprobs)))
    assert math.isclose(bottom_perp, math.exp(-sum(logprobs) / len(logprobs)))


def test_perplexity_metrics_identical_logprobs():
    logprobs = [-2.0] * 10
    perp, bottom_perp = compute_perplexity_metrics(logprobs, bottom_k=5)
    expected = math.exp(2.0)
    assert math.isclose(perp, expected)
    assert math.isclose(bottom_perp, expected)


def test_get_devices_slices():
    gpus_per_node = 8
    gpus_per_replica = 2
    slices = get_device_slices(gpus_per_node, gpus_per_replica)

    expected_slices = ["0,1", "2,3", "4,5", "6,7"]
    assert slices == expected_slices

    # Test with gpus_per_node not divisible by gpus_per_replica
    gpus_per_node = 9
    slices = get_device_slices(gpus_per_node, gpus_per_replica)
    expected_slices = ["0,1", "2,3", "4,5", "6,7", "8"]
    assert slices == expected_slices

    gpus_per_node = 4
    gpus_per_replica = 4
    slices = get_device_slices(gpus_per_node, gpus_per_replica)
    expected_slices = ["0,1,2,3"]
    assert slices == expected_slices
