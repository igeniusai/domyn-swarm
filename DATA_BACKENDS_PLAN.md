# Data Backends (Per-Job) — Design & Implementation Plan

This document is the design spec and tracking checklist for introducing **per-job**
data reading/writing backends across the swarm job pipeline.

**Primary drivers**
- Reduce hard coupling to pandas in the job runner pipeline.
- Support larger-than-memory or distributed datasets (Ray) and faster IO (Polars).
- Keep existing behavior stable for current users (pandas default).

**Hard constraints (confirmed)**
- Backend selection is **per job**.
- Persistence format is **parquet only** (for runner/checkpointing outputs).
- Ray support is **optional** and must be implemented with **lazy imports** and
  clear missing-dependency errors.
- CLI job submission must expose backend selection (and native backend mode)
  as convenience flags (even though `job_kwargs` already exist).

**Current touchpoints (where this will land)**
- `src/domyn_swarm/helpers/io.py`: pandas-only IO utilities.
- `src/domyn_swarm/jobs/runner.py`: unified runner that currently accepts `pd.DataFrame`.
- `src/domyn_swarm/checkpoint/store.py`: parquet shard checkpoint store (pandas + fsspec).
- `src/domyn_swarm/jobs/base.py`: `SwarmJob.transform_items(items: list[Any])` API contract.
- `src/domyn_swarm/cli/job.py`: CLI entry for job submission.

---

## 1) Implementation Plan

### 1.1 Scope, Semantics, and Non-Goals
Why: Avoid building an abstraction that accidentally changes job semantics (ordering, ids,
checkpoint correctness) or expands scope (new output formats, new checkpoint store types).

How:
- Explicitly define **row identity**, **ordering**, and **what must remain parquet-only**.
- Document what is *out of scope* for the first iteration (e.g., native write sharding
for ray/polars, new checkpoint formats).

- [ ] Task: Write a short "Semantics" section in the docs and code comments.
  - [ ] Subtask: Define `id_col` rules (default, generation, stability requirements).
  - [ ] Subtask: Define ordering semantics (results joined by `id_col`, not positional).
  - [ ] Subtask: Define what “parquet-only” means (checkpoint shards and final outputs).
  - [ ] Subtask: List non-goals (e.g., “ray writes sharded parquet” as future work).

Acceptance criteria:
- A reader can answer “how do outputs map to inputs?” and “what happens if ray isn’t installed?”
  without reading the code.

### 1.2 DataBackend Contract (Protocol) + Batch Model
Why: The contract determines what is possible across pandas/polars/ray. If this is wrong,
every downstream integration becomes awkward (especially native backend mode).

How:
- Define a minimal **DataBackend** protocol that supports:
  - read parquet
  - slice/shard
  - schema inspection
  - optional batch iteration for native mode
  - conversion to pandas for compatibility mode

Recommended design choices:
- Keep the protocol **small** and prefer adding utility helpers later.
- Treat `iter_batches` as **optional** (backend can raise a clear error if native mode is requested
  but not supported).
- Standardize the **batch object** shape for native mode to preserve checkpoint correctness.

Batch model proposal (native mode):
- Batches must carry:
  - ids (stable identifiers)
  - input payloads (one or more input columns)
- Output flush must still operate on `(ids, rows)` where `rows` are scalars/tuples/dicts.

- [ ] Task: Define `DataBackend` protocol and its docstring.
  - [ ] Subtask: Decide if `iter_batches` returns Python lists, an iterator, or async iterator.
  - [ ] Subtask: Define a `JobBatch` shape (e.g., `{id_col: ..., input_cols...}`) that works for
        pandas/polars/ray.
  - [ ] Subtask: Define schema normalization (stringified dtypes, column order rules).
  - [ ] Subtask: Decide how backend-specific IO kwargs are passed to `read()`/`write()`.

Acceptance criteria:
- The protocol can represent:
  - a pandas DataFrame
  - a polars DataFrame
  - a ray Dataset or batch dict
  without losing id mapping capability.

### 1.3 Backend Registry + Lazy Import Policy
Why: Optional dependencies must not be imported unless selected, otherwise environments without
ray/polars will break just by importing domyn-swarm.

How:
- Centralize resolution via a registry function (e.g., `get_backend("pandas")`).
- Perform lazy import inside resolver for polars/ray.
- Standardize missing-dependency exceptions and error messages.

Error messaging requirements:
- CLI should error early when possible (validate flags).
- Runner should error clearly if backend is requested but deps are missing.

- [ ] Task: Specify backend names and registry resolution behavior.
  - [ ] Subtask: Decide canonical names (`pandas`, `polars`, `ray`) and aliases (optional).
  - [ ] Subtask: Define exception types/messages (e.g., “Install `polars`” / “Install `ray[data]`”).
  - [ ] Subtask: Decide where validation happens (CLI parse vs runtime).

Acceptance criteria:
- Importing domyn-swarm does not import ray or polars unless explicitly selected.

### 1.4 Repo Restructuring (New Subpackage)
Why: The backend implementation should live in its own package to keep IO logic
separate from generic helpers and to reduce cross-module coupling.

How:
- Introduce `src/domyn_swarm/data/backends/` for the protocol, registry, and implementations.
- Keep `helpers/io.py` as the pandas-specific IO layer (wrapped by PandasBackend).
- Re-export a minimal API from `domyn_swarm.data` for consumers.

- [ ] Task: Create the `data/backends` subpackage and its `__init__.py`.
- [ ] Subtask: Move backend implementations under the new package.
- [ ] Subtask: Keep legacy helpers intact and call them from PandasBackend.

Acceptance criteria:
- Backend code is isolated from helpers and can evolve without touching unrelated modules.

### 1.5 Job Metadata (Per-Job Backend Selection) + Serialization
Why: “Per job” means the selection must survive:
- serialization (`SwarmJob.to_kwargs()`)
- remote reconstruction (`domyn_swarm.jobs.run`)
- CLI submission and rehydration paths

How:
- Add new fields to `SwarmJob` (or its constructor kwargs) that are JSON-serializable:
  - `data_backend: str | None` (default: `"pandas"`)
  - `native_backend: bool` (default: `False`)
  - (optional) `native_batch_size: int` if needed for consistent behavior

Important constraint:
- `SwarmJob.to_kwargs()` currently serializes `__dict__` fields that are basic JSON types.
  Any new fields must fit that filter.

- [ ] Task: Specify job fields and their defaults.
  - [ ] Subtask: Decide naming (`data_backend` vs `backend`) to avoid collisions with existing
        “backend” concepts (serving/compute backends already exist in the repo).
  - [ ] Subtask: Specify how CLI flags map to these fields (override precedence).
  - [ ] Subtask: Define backward-compat behavior when fields are absent.

Acceptance criteria:
- A job created on one machine and executed in-cluster still uses the intended backend.

### 1.6 Runner Wiring (Compatibility Mode)
Why: We need backend selection without breaking the existing `SwarmJob.transform_items(items: list[Any])`
contract and without rewriting checkpoint logic immediately.

How:
- Add backend resolution at the runner boundary (likely in `domyn_swarm.jobs.run` and/or
  `JobRunner.run`).
- In compatibility mode (`native_backend=False`):
  - read data via backend
  - convert to pandas (`to_pandas()`)
  - keep current logic: extract items as Python list, call `transform_streaming`, flush to parquet store

- [ ] Task: Define compatibility-mode flow explicitly (with pseudo-code in docs).
  - [ ] Subtask: Ensure id_col generation happens consistently (and before checkpointing).
  - [ ] Subtask: Ensure `limit` behavior is consistent across backends.
  - [ ] Subtask: Ensure mixed input_col forms (string vs list/tuple) are handled.

Acceptance criteria:
- With `data_backend="pandas"` and `native_backend=False`, behavior is identical to today.

### 1.7 CLI: `job submit` Flags + Mapping
Why: While `job_kwargs` can carry arbitrary data, users need a simple UX:
- pick backend with a flag
- optionally enable native mode
- optionally set batch size

How:
- Add flags to `domyn-swarm job submit` (in `src/domyn_swarm/cli/job.py`):
  - `--data-backend {pandas,polars,ray}` (or `--backend`, but avoid ambiguity)
  - `--native-backend / --no-native-backend`
  - `--native-batch-size <int>` (optional; only meaningful if native-backend)

Mapping rules (proposal):
- CLI flags override values embedded in `job_kwargs`, because flags are explicit operator intent.
- If neither is provided, default to pandas + non-native.

- [ ] Task: Specify CLI UX and precedence rules.
  - [ ] Subtask: Confirm flag names (avoid collision with swarm compute/serving backends).
  - [ ] Subtask: Specify validation rules (e.g., native-backend requires data-backend != pandas? or allow).
  - [ ] Subtask: Ensure CLI help includes optional dependency guidance.
  - [ ] Subtask: Add CLI passthrough for backend IO kwargs (e.g., `--backend-read-kwargs`).

Acceptance criteria:
- A user can run: `domyn-swarm job submit --data-backend polars --native-backend ...`
  without writing custom job code.

### 1.8 Parquet-Only Persistence Alignment
Why: Cross-backend consistency depends on writing the same storage format for:
- checkpoint parts
- final outputs

How:
- Keep `CheckpointStore` parquet shard store as the single persistence mechanism.
- Ensure any “backend write” is used only for input reading and for producing the final parquet output
  (if/when the final output is written outside checkpoint store).

Policy decisions to make explicit:
- Pandas-only CSV/JSONL IO: keep or deprecate? (The broader system is parquet-centric.)
- Non-pandas backends: parquet-only enforced from day 1.

- [ ] Task: Decide and document parquet-only write enforcement rules.
  - [ ] Subtask: Decide whether pandas retains CSV/JSONL reads (possibly as debug-only).
  - [ ] Subtask: Ensure all non-parquet writes raise a clear error.
  - [ ] Subtask: Confirm output path conventions for shards vs final output.

Acceptance criteria:
- All backends converge on producing a parquet dataset + final parquet artifact.

### 1.9 Native Backend Mode (Performance Path)
Why: Converting everything to pandas can be slow or impossible for large datasets.
Native backend mode is the opt-in path for:
- polars native batching
- ray distributed dataset batching

How:
- Add an additional method to `SwarmJob` (opt-in):
  - `transform_items_native(batch, *, backend_name: str, id_col: str, input_cols: list[str])`
- Runner behavior:
  - if `native_backend=True`, runner uses `backend.iter_batches(...)`
  - each batch must include ids and inputs
  - job returns results aligned to the batch ids
  - flush uses the same `(ids, rows)` mechanism as today

Key design requirement:
- Native mode must not weaken checkpoint correctness. That means ids must be stable,
  and batch output must map back to ids exactly.

- [ ] Task: Specify native method signature and semantics.
  - [ ] Subtask: Define how ids are attached to batches for each backend.
  - [ ] Subtask: Define expected output shapes (scalar, tuple, dict) and normalization.
  - [ ] Subtask: Define behavior when native mode is requested but job doesn’t implement the native method.
  - [ ] Subtask: Decide where batch_size lives (job config vs CLI override).

Acceptance criteria:
- Native mode is fully optional and does not impact default behavior.
- When enabled, the job can process data without a full conversion to pandas.

#### 1.9.1 Ray Fault Tolerance and “Output-As-Checkpoint” Strategy (Investigation)
Why: A “true” Ray backend should distribute LLM calls across Ray workers, but we also need a
fault-tolerance story before we invest in a custom distributed checkpoint store.

How (proposal, to validate):
- Use Ray’s task-level retry and Ray Data execution resilience for worker/node failures.
- Accept **at-least-once** processing semantics at the compute layer (tasks may be retried).
- Preserve correctness by making outputs **id-addressed**:
  - Every input row has a stable `id_col`.
  - Every output row includes that `id_col`.
  - Final materialization (or merge) de-duplicates by `id_col` (“last write wins”).
- For “resume” without a bespoke store, treat the output directory as a checkpoint:
  1) On start, if output shards already exist, read them to compute `done_ids`.
  2) Filter the input dataset to rows where `id_col` not in `done_ids`.
  3) Continue writing additional output shards.

Important nuance (LLM side effects):
- Even if compute is retried safely, LLM requests are not necessarily idempotent.
  “At-least-once” means duplicate requests may happen on retries; de-dup prevents duplicate
  *rows* but does not prevent extra LLM calls. We should explicitly document this.

- [ ] Task: Validate Ray Data fault tolerance guarantees relevant to this pipeline.
  - [ ] Subtask: Confirm retry behavior for `map_batches` / dataset transforms on worker failure.
  - [ ] Subtask: Confirm how partial parquet writes behave on failure and what remains readable.
  - [ ] Subtask: Decide and document processing semantics (at-least-once vs exactly-once).
- [ ] Task: Define a Ray-native “output-as-checkpoint” resume algorithm and constraints.
  - [ ] Subtask: Specify output layout (`output_dir/part-*.parquet`) and how to detect done ids.
  - [ ] Subtask: Specify de-dup rule (by `id_col`) and where it occurs (finalize step).
  - [ ] Subtask: Decide how to handle partially written shards or corrupted parts.

### 1.10 Backend Implementations (pandas → polars → ray)
Why: Implement in increasing complexity and risk; pandas is the baseline.

How:
- Pandas backend:
  - delegate to existing `helpers/io.py` in the first iteration
- Polars backend:
  - read parquet via polars
  - batches are polars DataFrames or dicts
- Ray backend:
  - read parquet via ray.data (lazy import)
  - batches are dicts (arrow/pandas blocks) or ray batch objects

- [ ] Task: Define per-backend behavior (explicitly).
  - [ ] Subtask: Pandas backend: confirm directory parquet datasets and glob patterns.
  - [ ] Subtask: Polars backend: confirm how to implement `limit` efficiently.
  - [ ] Subtask: Ray backend: confirm order/partition semantics; define requirements on id_col presence.
  - [ ] Subtask: Define schema normalization outputs (string types, stable mapping).

Acceptance criteria:
- Each backend can read parquet and provide batches that include id + input columns.

### 1.11 Tests, Rollout, and Documentation
Why: This crosses the job execution boundary and can silently break correctness if untested.

How:
- Unit tests:
  - registry lazy import behavior
  - error messages for missing optional deps
  - CLI flag parsing/mapping
- Integration tests:
  - a simple job that echoes inputs, across backends (skip polars/ray tests when deps missing)
  - native_backend mode branching behavior

- [ ] Task: Define the test matrix and skip behavior.
  - [ ] Subtask: Decide what runs in CI by default (likely pandas-only).
  - [ ] Subtask: Add optional CI job (or local instructions) for polars/ray coverage.
  - [ ] Subtask: Add docs: “Selecting data backend” and “Native backend mode”.

Acceptance criteria:
- CI proves pandas path unchanged.
- Optional backends are covered by at least unit tests for missing deps + basic behaviors.

---

## 2) Draft API and Flow (Illustrative, Not Final Code)

### 2.1 Proposed CLI UX
- `domyn-swarm job submit --data-backend pandas ...` (default)
- `domyn-swarm job submit --data-backend polars ...`
- `domyn-swarm job submit --data-backend ray ...`
- `domyn-swarm job submit --data-backend ray --native-backend --native-batch-size 1024 ...`

### 2.2 Compatibility-mode flow (today’s semantics preserved)
1) Read parquet input
2) Convert to pandas
3) Extract list of items from input column(s)
4) Call `SwarmJob.transform_streaming(items, ...)`
5) Flush outputs to parquet checkpoint store
6) Finalize into output parquet

### 2.3 Backend IO Parameters (Read/Write Kwargs)
Why: Different IO engines require different options (e.g., fsspec `storage_options`,
row-group reads, columns projection, ray filesystem config). These must be configurable
without hardcoding per backend.

How:
- Allow passing backend-specific kwargs into backend `read()`/`write()` calls.
- Prefer a structured, JSON-serializable mechanism so it can flow through `job_kwargs`
  and CLI.

Proposed approach (design):
- Job-level fields:
  - `backend_read_kwargs: dict[str, Any] | None`
  - `backend_write_kwargs: dict[str, Any] | None`
- CLI:
  - `--backend-read-kwargs '<json>'`
  - `--backend-write-kwargs '<json>'`
- Precedence:
  - CLI kwargs override job-provided kwargs (explicit operator intent).

Tracking tasks:
- [ ] Add job-level fields for `backend_read_kwargs` / `backend_write_kwargs`.
- [ ] Define validation (JSON parse errors, unknown keys allowed).
- [ ] Ensure kwargs remain JSON-serializable for remote execution.

### 2.4 Native-mode flow (opt-in)
1) Read parquet input via backend
2) Iterate backend-native batches that include id + input columns
3) Call `SwarmJob.transform_items_native(batch, ...)`
4) Flush outputs to parquet checkpoint store by id
5) Finalize into output parquet

---

## 3) Future API: `transform_item` (Preferred for New Jobs)

Why: The current privileged API is `transform_items(items: list[Any])`, but many jobs are
conceptually “map one input to one output”. A single-item API:
- reduces boilerplate (no list wrapping/unwrapping)
- makes semantics clearer (one input → one output)
- is a better fit for streaming and native-batch execution (runner can own batching)

How:
- Introduce an optional `transform_item(item)` coroutine on `SwarmJob`.
- The framework continues to support `transform_items` for backward compatibility.
- Runner chooses the best available method in priority order:
  1) `transform_item` (new)
  2) `transform_items` (existing)

Proposed method signature:
```
async def transform_item(self, item: Any) -> Any:
    ...
```

Compatibility plan:
- Default implementation can be a shim that calls `transform_items([item])` and returns the first
  element, preserving existing jobs unchanged.
- Long-term: documentation and templates encourage implementing `transform_item` for new jobs.

Interaction with native backend mode:
- In compatibility mode, runner can call `transform_item` over a Python list of items.
- In native backend mode, `transform_items_native` remains the dedicated extension point for
  backend-native batches; it can internally call `transform_item` if a job wants to share logic.

Tracking tasks (future):
- [ ] Add `transform_item` optional method to `SwarmJob` and document precedence rules.
- [ ] Update runner to prefer `transform_item` when present.
- [ ] Add tests ensuring equivalence vs `transform_items` (ordering, errors, retries).
