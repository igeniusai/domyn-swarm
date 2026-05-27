# Domyn Swarm

Domyn-Swarm is a CLI + Python library for launching **LLM serving endpoints** (vLLM, OpenAI-compatible)
on **Slurm** or **DGX Cloud Lepton**, and for running **high-throughput batch jobs** against those endpoints
with retries + checkpointing.

## Quick start
- Install deps (includes optional extras): `uv sync --all-extras`
- Run the CLI locally: `uv run domyn-swarm --help`
- Run a focused test: `uv run pytest tests/cli/test_main.py -q`
- Create a defaults file for your environment: `uv run domyn-swarm init defaults`
- Implementation plans and tasks are stored in the .plans folder

## Common commands
- Tests (focused first): `uv run pytest tests/ -q`
- Tests (full suite): `uv run pytest`
- Tests (skip integration): `uv run pytest -m "not integration"`
- Tests (integration only): `uv run pytest -m integration -v`
- Format: `uv run ruff format .`
- Lint and fix with ruff: `uv run ruff check --fix .`
- Lint: `uv run ruff check .`
- Type check: `uv run pyright`
- Pre-commit (optional): `uv run pre-commit run --all-files`

## Development workflow
- After a non-trivial change: run the most relevant test file(s) first, then `uv run pytest` to catch regressions.
- Keep changes inside `src/` and `tests/`; avoid committing generated artifacts under `output/`, `htmlcov/`, `.pytest_cache/`, `.ruff_cache/`, `.domyn_swarm/`, and `.checkpoints/`.
- Before committing: run `uv run pre-commit run --all-files` and ensure all hooks pass.
- Don’t edit `uv.lock` unless you intentionally changed dependencies.
- Use conventional commits when generating commits for this project
- Always add docstrings in Google format to functions, methods and classes you define in the src/ directory. Do not put docstrings in test functions unless needed for clarity

## Codebase map
### Entry points
- CLI app: `src/domyn_swarm/cli/main.py` (Typer app; auto-upgrades DB on startup unless `DOMYN_SWARM_SKIP_DB_UPGRADE=1`)
- Programmatic API: `src/domyn_swarm/core/swarm.py` (`DomynLLMSwarm` context manager)
- In-cluster job runner entry: `src/domyn_swarm/jobs/cli/run.py` (`python -m domyn_swarm.jobs.cli.run ...`)

### Core orchestration
- Swarm lifecycle + job submission: `src/domyn_swarm/core/swarm.py`
- Deployment abstraction (ServingBackend + ComputeBackend): `src/domyn_swarm/deploy/deployment.py`
- Platform protocols / status types: `src/domyn_swarm/platform/protocols.py`

### Backends (platform integration)
- Slurm serving: `src/domyn_swarm/backends/serving/slurm.py`, `src/domyn_swarm/backends/serving/slurm_driver.py`
- Slurm readiness + status normalization: `src/domyn_swarm/backends/serving/slurm_readiness.py`
- Slurm compute (`srun` wrapper): `src/domyn_swarm/backends/compute/slurm.py` (builds commands via `src/domyn_swarm/backends/serving/srun_builder.py`)
- Lepton integration (optional extra): `src/domyn_swarm/backends/serving/lepton.py`, `src/domyn_swarm/backends/compute/lepton.py`

### Jobs (batch inference drivers)
- Base API + OpenAI client wiring: `src/domyn_swarm/jobs/base.py` (`SwarmJob`)
- Built-in job types: `src/domyn_swarm/jobs/chat_completion.py` (excluded from Pyright for now)
- Runner + sharding + checkpoint-store integration: `src/domyn_swarm/jobs/runner.py`
- Concurrency + retry batching helper: `src/domyn_swarm/jobs/batching.py`
- Legacy checkpointing (Parquet + file lock): `src/domyn_swarm/checkpoint/manager.py`
- New checkpoint store (sharded parquet): `src/domyn_swarm/checkpoint/store.py`

### Config
- Main YAML schema: `src/domyn_swarm/config/swarm.py` (`DomynLLMSwarmConfig`)
- Backend configs + plan builder: `src/domyn_swarm/config/backend.py`, `src/domyn_swarm/config/plan.py`
- Environment settings (.env + env vars): `src/domyn_swarm/config/settings.py`
- Defaults file loader (search order + cache): `src/domyn_swarm/config/defaults.py`

### State & persistence (SQLite)
- Local state DB (`swarm.db`) schema + CRUD: `src/domyn_swarm/core/state/state_manager.py`
- Alembic migrations + auto-upgrade: `src/domyn_swarm/core/state/migrate.py`, `src/domyn_swarm/core/state/autoupgrade.py`
- Watchdog status DB (`watchdog.db`) helpers: `src/domyn_swarm/core/state/watchdog.py`

### Runtime health (watchdog + collector)
- Per-replica watchdog process: `src/domyn_swarm/runtime/watchdog.py`
- Collector (single writer to `watchdog.db`): `src/domyn_swarm/runtime/collector.py`
- Reads replica status for `domyn-swarm status`: `src/domyn_swarm/runtime/status.py`

### UI (terminal)
- TUI rendering + components: `src/domyn_swarm/cli/tui/` (used by `domyn-swarm status` and list views)

### Data backends (Parquet I/O)
- Backend registry: `src/domyn_swarm/data/backends/registry.py`
- Built-ins: pandas (default), polars (optional), ray (optional)

## Style and conventions
### Tooling defaults
- Formatting/lint: Ruff (`line-length=100`, target `py310`), configured in `pyproject.toml`
- Type checking: Pyright, configured in `pyproject.toml` (note the explicit excludes)
- Tests: Pytest (+ pytest-asyncio), with `integration` marker registered in `pyproject.toml`

### Repo-specific conventions
- Prefer adding new code under `src/domyn_swarm/` and tests under `tests/`.
- If you add a new CLI flag or config field, update both:
  - schema/validation (`src/domyn_swarm/config/`)
  - CLI plumbing (`src/domyn_swarm/cli/`)
- When touching state DB models or behavior, add/adjust Alembic migrations (`src/domyn_swarm/core/state/migrations/`).

## Notes on dependencies and jobs
### Key runtime dependencies
- LLM client: `openai` python SDK (configured to talk to a vLLM OpenAI-compatible server via `base_url=ENDPOINT`)
- Orchestration + CLI: `typer`, `rich`, `pydantic` v2
- Persistence: `sqlalchemy` + `alembic` (local `swarm.db`), plus per-swarm `watchdog.db`
- Data I/O: `pandas` + `pyarrow` (default), optional `polars`, optional `ray`

### Optional extras
- Lepton support: install `domyn-swarm[lepton]` (or `uv sync --all-extras`) and set `LEPTONAI_API_TOKEN` / `LEPTON_WORKSPACE_ID`.

### Where state lives (important for debugging)
- Global state DB: `${DOMYN_SWARM_HOME:-~/.domyn_swarm}/swarm.db`
- Per-swarm directory: `${DOMYN_SWARM_HOME:-~/.domyn_swarm}/swarms/<swarm-name>/`
- Per-swarm health DB: `.../swarms/<swarm-name>/watchdog.db`
- Defaults file search order (first hit wins): `DOMYN_SWARM_DEFAULTS` → `./defaults.yaml` → `./.domyn_swarm/defaults.yaml` → `~/.domyn_swarm/defaults.yaml`

### Environment variables you’ll see in code/tests
- `.env` is loaded automatically from the repo root (CWD) and `~/.domyn_swarm/.env` (see `src/domyn_swarm/config/settings.py`).
- User-facing settings: `DOMYN_SWARM_HOME`, `DOMYN_SWARM_LOG_LEVEL`, `DOMYN_SWARM_DEFAULTS`, `DOMYN_SWARM_SKIP_DB_UPGRADE`, `DOMYN_SWARM_ASCII`
- Tokens/aliases: `DOMYN_SWARM_API_TOKEN`, `VLLM_API_KEY`, `SINGULARITYENV_VLLM_API_KEY`, `LEPTONAI_API_TOKEN`, `LEPTON_WORKSPACE_ID`
- In-cluster job runner env (injected by the platform/driver): `ENDPOINT`, `MODEL`, `JOB_CLASS`, `INPUT_PARQUET`, `OUTPUT_PARQUET`, `JOB_KWARGS`

### Tests: what’s “integration” here?
- Integration tests are marked `@pytest.mark.integration` and live mainly under `tests/runtime/`.
- These are still self-contained (they spawn local subprocesses like the watchdog + a fake ray CLI), but they run slower; prefer `-m "not integration"` for tight loops.
