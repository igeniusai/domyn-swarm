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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from domyn_swarm.checkpoint.arrow_store import ArrowShardStore, InMemoryArrowStore
from domyn_swarm.jobs.arrow_runner import ArrowJobRunner, ArrowRunnerConfig
from domyn_swarm.jobs.base import OutputJoinMode, SwarmJob

if TYPE_CHECKING:
    import polars as pl


@dataclass
class PolarsRunnerConfig:
    """Configuration for PolarsJobRunner."""

    id_col: str = "_row_id"
    checkpoint_every: int = 16


class PolarsJobRunner:
    """Polars-native runner that delegates execution to ArrowJobRunner."""

    def __init__(
        self, store: ArrowShardStore | InMemoryArrowStore, cfg: PolarsRunnerConfig | None = None
    ):
        """Initialize the runner with a checkpoint store and config."""
        self.store = store
        self.cfg = cfg or PolarsRunnerConfig()

    def _materialize(self, data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
        """Materialize a LazyFrame to a DataFrame when needed.

        Args:
            data: Polars DataFrame or LazyFrame.

        Returns:
            Materialized Polars DataFrame.
        """
        import polars as pl

        if isinstance(data, pl.LazyFrame):
            return data.collect()
        return data

    async def run(
        self,
        job: SwarmJob,
        data: pl.DataFrame | pl.LazyFrame,
        *,
        input_col: str,
        output_cols: list[str] | None,
        output_mode: OutputJoinMode | None = None,
    ) -> pl.DataFrame:
        """Run a SwarmJob against a polars DataFrame or LazyFrame.

        Args:
            job: SwarmJob instance to execute.
            data: Input polars DataFrame or LazyFrame.
            input_col: Column name to read inputs from.
            output_cols: Output column names (None for dict outputs).
            output_mode: Output join mode (APPEND, REPLACE, IO_ONLY).

        Returns:
            Polars DataFrame with job outputs.
        """
        import polars as pl

        df = self._materialize(data)
        table = df.to_arrow()
        arrow_cfg = ArrowRunnerConfig(
            id_col=self.cfg.id_col,
            checkpoint_every=self.cfg.checkpoint_every,
        )
        runner = ArrowJobRunner(self.store, arrow_cfg)
        out_table = await runner.run(
            job,
            table,
            input_col=input_col,
            output_cols=output_cols,
            output_mode=output_mode,
        )
        return cast(pl.DataFrame, pl.from_arrow(out_table))


async def run_polars_job(
    job_factory: Any,
    data: pl.DataFrame | pl.LazyFrame,
    *,
    input_col: str,
    output_cols: list[str] | None,
    store_uri: str | None,
    checkpoint_every: int,
    checkpointing: bool,
    id_col: str,
) -> pl.DataFrame:
    """Run a job using the polars runner and Arrow checkpointing.

    Args:
        job_factory: Callable returning a SwarmJob instance.
        data: Input polars DataFrame or LazyFrame.
        input_col: Input column name.
        output_cols: Output column names (None for dict outputs).
        store_uri: Base checkpoint store URI (required if checkpointing).
        checkpoint_every: Flush interval in items.
        checkpointing: Whether to read/write checkpoint state.
        id_col: Column name used for stable row ids.

    Returns:
        Polars DataFrame containing job outputs.
    """
    if checkpointing and store_uri is None:
        raise ValueError("store_uri is required when checkpointing is enabled.")

    store = ArrowShardStore(store_uri) if checkpointing and store_uri else InMemoryArrowStore()
    cfg = PolarsRunnerConfig(
        id_col=id_col,
        checkpoint_every=checkpoint_every,
    )
    runner = PolarsJobRunner(store, cfg)
    job = job_factory()
    return await runner.run(
        job,
        data,
        input_col=input_col,
        output_cols=output_cols,
        output_mode=getattr(job, "output_mode", OutputJoinMode.APPEND),
    )
