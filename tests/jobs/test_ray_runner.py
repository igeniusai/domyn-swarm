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

from __future__ import annotations

from pathlib import Path
import sys
import types
from typing import Any

import pandas as pd
import pytest

from domyn_swarm.jobs.ray_runner import _ensure_ray_initialized, run_ray_job


class _FakeDataset:
    """A tiny Ray Dataset-like object for unit tests.

    This is intentionally minimal: it supports the methods used by `run_ray_job`.
    """

    def __init__(self, *, batches: list[pd.DataFrame]):
        self._batches = batches
        self.map_batches_calls: list[dict[str, Any]] = []
        self.write_parquet_calls: list[str] = []

    def map_batches(self, fn, *, batch_format: str, batch_size: int):
        self.map_batches_calls.append(
            {"batch_format": batch_format, "batch_size": batch_size, "fn": fn}
        )
        out_batches = [fn(b) for b in self._batches]
        return _FakeDataset(batches=out_batches)

    def write_parquet(self, path: str) -> None:
        self.write_parquet_calls.append(path)
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "_SUCCESS").write_text("ok")


class _FakeReadDataset:
    """A tiny read_parquet result that supports take_all()."""

    def __init__(self, rows: list[dict[str, Any]]):
        self._rows = rows

    def take_all(self) -> list[dict[str, Any]]:
        return list(self._rows)


def _install_fake_ray(monkeypatch: pytest.MonkeyPatch, *, initialized: bool = False):
    """Install a fake `ray` + `ray.data` in sys.modules.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        initialized: Whether `ray.is_initialized()` should return True initially.

    Returns:
        Tuple of (ray_module, ray_data_module) for assertions.
    """
    ray_mod = types.ModuleType("ray")
    ray_state = {"initialized": initialized, "init_calls": []}

    def is_initialized() -> bool:
        return bool(ray_state["initialized"])

    def init(*, address: str, ignore_reinit_error: bool, log_to_driver: bool) -> None:
        ray_state["init_calls"].append(
            {
                "address": address,
                "ignore_reinit_error": ignore_reinit_error,
                "log_to_driver": log_to_driver,
            }
        )
        ray_state["initialized"] = True

    ray_mod.is_initialized = is_initialized  # type: ignore[attr-defined]
    ray_mod.init = init  # type: ignore[attr-defined]

    rd_mod = types.ModuleType("ray.data")
    rd_state = {"read_parquet_calls": [], "from_items_calls": []}

    def read_parquet(paths, *, columns=None):
        rd_state["read_parquet_calls"].append({"paths": list(paths), "columns": columns})
        if columns:
            return _FakeReadDataset(rows=[{columns[0]: 1}])
        return _FakeDataset(batches=[])

    def from_items(items):
        rd_state["from_items_calls"].append(list(items))
        return _FakeDataset(batches=[])

    rd_mod.read_parquet = read_parquet  # type: ignore[attr-defined]
    rd_mod.from_items = from_items  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "ray", ray_mod)
    monkeypatch.setitem(sys.modules, "ray.data", rd_mod)
    return ray_mod, rd_mod, ray_state, rd_state


def test_ensure_ray_initialized_requires_address(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise if ray is not initialized and no address is provided."""
    ray_mod, _rd_mod, _ray_state, _rd_state = _install_fake_ray(monkeypatch, initialized=False)
    monkeypatch.delenv("DOMYN_SWARM_RAY_ADDRESS", raising=False)
    monkeypatch.delenv("RAY_ADDRESS", raising=False)

    with pytest.raises(ValueError, match="explicit ray address"):
        _ensure_ray_initialized(ray_mod, ray_address=None)


def test_ensure_ray_initialized_connects_to_cluster(monkeypatch: pytest.MonkeyPatch) -> None:
    """Call ray.init(address=...) when address is provided."""
    ray_mod, _rd_mod, ray_state, _rd_state = _install_fake_ray(monkeypatch, initialized=False)
    _ensure_ray_initialized(ray_mod, ray_address="ray://head:10001")
    assert ray_state["initialized"] is True
    assert ray_state["init_calls"][0]["address"] == "ray://head:10001"


@pytest.mark.asyncio
async def test_run_ray_job_checkpointing_requires_store_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise if checkpointing is enabled but store_uri is missing."""
    _install_fake_ray(monkeypatch, initialized=False)
    monkeypatch.setenv("DOMYN_SWARM_RAY_ADDRESS", "ray://head:10001")

    class _Job:
        async def transform_items(self, items: list[Any]) -> list[Any]:
            return items

    ds = _FakeDataset(batches=[pd.DataFrame({"doc_id": [1], "messages": ["x"]})])
    with pytest.raises(ValueError, match="store_uri is required"):
        await run_ray_job(
            _Job,
            ds,
            input_col="messages",
            output_cols=["out"],
            batch_size=8,
            output_mode="append",
            id_col="doc_id",
            store_uri=None,
            checkpointing=True,
            compact=True,
            ray_address=None,
        )


@pytest.mark.asyncio
async def test_run_ray_job_filters_done_ids_and_writes_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Resume filters done ids and writes a new run directory, then compacts outputs."""
    _install_fake_ray(monkeypatch, initialized=False)
    monkeypatch.setenv("DOMYN_SWARM_RAY_ADDRESS", "ray://head:10001")

    class _Job:
        def transform_items(self, items: list[Any]) -> list[Any]:
            return [f"ok-{x}" for x in items]

    base_df = pd.DataFrame({"doc_id": [1, 2], "messages": ["a", "b"]})
    ds = _FakeDataset(batches=[base_df])

    # Pre-create an existing run dir so done_ids loader sees something.
    runs_base = tmp_path / "ckp"
    (runs_base / "runs" / "run-old").mkdir(parents=True)
    store_uri = f"file://{runs_base / 'store.parquet'}"

    out = await run_ray_job(
        _Job,
        ds,
        input_col="messages",
        output_cols=["out"],
        batch_size=8,
        output_mode="io_only",
        id_col="doc_id",
        store_uri=store_uri,
        checkpointing=True,
        compact=True,
        ray_address="ray://head:10001",
    )

    # Compaction returns a dataset-like object from rd.read_parquet (fake returns _FakeDataset).
    assert isinstance(out, _FakeDataset)

    # A new run directory was written (in addition to existing run-old).
    runs_dir = runs_base / "store" / "runs"
    assert runs_dir.exists()
    run_dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    assert any(p.name.startswith("run-") for p in run_dirs)
