# Domyn Swarm

Domyn Swarm is a CLI and Python library for launching OpenAI-compatible LLM
serving endpoints on Slurm or DGX Cloud Lepton and running high-throughput batch
jobs against them with retries and checkpointing.

## Quick start

- Install development and optional dependencies: `uv sync --all-extras`.
- Inspect the CLI: `uv run domyn-swarm --help`.
- Create environment defaults: `uv run domyn-swarm init defaults`.
- Run a focused test: `uv run pytest tests/cli/test_main.py -q --no-cov`.

## Development workflow

- Keep each change focused and preserve behavior unless the task explicitly
  changes it. Keep public imports, serialized forms, CLI output, templates, and
  persistence formats stable.
- Inspect the owning module, its callers, and its tests before editing. Reuse an
  existing abstraction when it already owns the rule being changed.
- Run the closest tests first. Run the full suite before review for any
  non-trivial change.
- Do not edit `uv.lock` unless dependencies changed intentionally.
- Do not commit generated or local artifacts under `output/`, `htmlcov/`,
  `.pytest_cache/`, `.ruff_cache/`, `.domyn_swarm/`, or `.checkpoints/`.
- Use conventional commit messages. Sign and sign off commits with
  `git commit -sS`.

## Architecture map

### Entry points

- CLI application: `src/domyn_swarm/cli/main.py`.
- Programmatic lifecycle API: `src/domyn_swarm/core/swarm.py`
  (`DomynLLMSwarm`).
- In-cluster job runner: `src/domyn_swarm/jobs/cli/run.py`.
- Supported package-root API: `src/domyn_swarm/__init__.py`.

### Core orchestration

- Swarm lifecycle and job submission: `src/domyn_swarm/core/swarm.py`.
- Run specification: `src/domyn_swarm/core/job_run.py`.
- Serving/compute composition: `src/domyn_swarm/deploy/deployment.py`.
- Platform protocols and normalized status types:
  `src/domyn_swarm/platform/protocols.py`.

### Platform backends

- Slurm serving and driver: `src/domyn_swarm/backends/serving/slurm.py` and
  `src/domyn_swarm/backends/serving/slurm_driver.py`.
- Slurm readiness: `src/domyn_swarm/backends/serving/slurm_readiness.py`.
- Slurm compute and command construction: `src/domyn_swarm/backends/compute/`
  and `src/domyn_swarm/backends/serving/srun_builder.py`.
- Lepton serving and compute: `src/domyn_swarm/backends/serving/lepton.py` and
  `src/domyn_swarm/backends/compute/lepton.py`.

### Jobs

- Authoring API and job configuration: `src/domyn_swarm/jobs/api/`.
- Engine-neutral execution: `src/domyn_swarm/jobs/execution/`.
- Checkpoint and column I/O policy: `src/domyn_swarm/jobs/io/`.
- In-cluster entry point: `src/domyn_swarm/jobs/cli/run.py`.
- Compatibility shims only: `src/domyn_swarm/jobs/base.py`,
  `src/domyn_swarm/jobs/runner.py`, `src/domyn_swarm/jobs/batching.py`, and
  `src/domyn_swarm/jobs/chat_completion.py`.
- Checkpoint stores: `src/domyn_swarm/checkpoint/store.py`; the legacy
  file-locking manager remains in `src/domyn_swarm/checkpoint/manager.py`.

### Configuration and persistence

- Main YAML schema: `src/domyn_swarm/config/swarm.py`.
- Backend schemas and deployment plan builder: `src/domyn_swarm/config/`.
- Submit-time filesystem checks: `src/domyn_swarm/config/preflight.py`.
- Local state schema and CRUD: `src/domyn_swarm/core/state/`.
- Alembic migrations:
  `src/domyn_swarm/core/state/migrations/`.

### Runtime health and terminal UI

- Replica watchdog: `src/domyn_swarm/runtime/watchdog.py`.
- Health collector and load-balancer supervisor: `src/domyn_swarm/runtime/`.
- Terminal views: `src/domyn_swarm/cli/tui/`.
- Data backend registry and adapters: `src/domyn_swarm/data/backends/`.

## Repository constraints

- Keep optional Lepton, Polars, and Ray imports lazy wherever the surrounding
  module supports installations without those extras.
- `src/domyn_swarm/runtime/watchdog.py` is bind-mounted into the serving
  container and must remain standard-library-only. Do not import
  `domyn_swarm` or third-party packages there.
- Treat names exported by `src/domyn_swarm/__init__.py` as the supported Python
  API. When adding or moving an export, update the targeted public-docstring
  hook in `.pre-commit-config.yaml`.
- Keep top-level job compatibility shims working until a documented removal.
- Pydantic `Field(description=...)` text is user-facing and feeds the generated
  configuration reference. Add or update it with every config field.
- Update both schema/validation under `src/domyn_swarm/config/` and CLI plumbing
  under `src/domyn_swarm/cli/` when adding a CLI option or configuration field.
- Add or adjust an Alembic migration for state database schema changes.
- Cover behavior-sensitive Jinja template changes with rendered-output tests.

## Code, documentation, and prose

### Design

- Prefer direct data flow, existing abstractions, and domain-specific names.
  Extract shared code only when callers share a real rule now.
- Keep functions and modules focused. Use early returns and named intermediate
  values when they make control flow easier to follow.
- Do not introduce generic managers, registries, configuration switches, or
  extension points for hypothetical use cases.
- Remove duplication in business rules, not merely text that happens to look
  similar.

### Docstrings

- Supported public APIs and non-obvious internal contracts need Google-style
  docstrings. Obvious private helpers, overrides, callbacks, validators,
  properties, and special methods do not.
- Start with a concise summary ending in punctuation. Add context for
  invariants, lifecycle, side effects, or compatibility constraints.
- Use `Args`, `Returns`, `Yields`, and `Raises` only when they add information
  beyond names and type annotations. Do not repeat types, defaults, or obvious
  implementation steps.
- Use test docstrings only for a behavior contract, regression history, or
  fixture constraint that the test name cannot express.

### Comments

- Comments are exceptional. Use one only when unusual code, a hidden invariant,
  an external constraint, a non-local side effect, or a non-obvious tradeoff
  would otherwise surprise a careful reader. Do not narrate straightforward
  code.
- Prefer clearer naming, types, and control flow over an explanatory comment.
- Keep necessary comments concise and explain why the code has its present
  shape. Remove a comment when its rationale no longer applies.
- Do not leave commented-out code, speculative notes, or instructions to future
  readers. Link actionable follow-up markers to an issue.
- Name the diagnostic in suppression comments and state the reason when it is
  not evident from the adjacent code.

### Project prose

- Use concise active voice, US English, and sentence-case headings.
- Use “Domyn Swarm” for the project, `domyn-swarm` for the CLI and distribution,
  and `domyn_swarm` for the Python package.
- Describe concrete behavior and consequences. Avoid promotional adjectives,
  stale temporal language, and second-person phrasing when a direct instruction
  is clearer.

## Verification

- Focused tests: `uv run pytest <test-path> -q --no-cov`.
- Full tests: `uv run pytest`.
- Tests without integration cases: `uv run pytest -m "not integration"`.
- Integration tests: `uv run pytest -m integration -v`.
- Format: `uv run ruff format .`.
- Check formatting: `uv run ruff format --check .`.
- Lint: `uv run ruff check .`.
- Type check: `uv run pyright`.
- Documentation: `uv run sphinx-build -W --keep-going -b html docs <output>`.
- All configured hooks: `uv run pre-commit run --all-files`.

Run the relevant focused checks while iterating. Before review, run the full
suite and pre-commit hooks unless the change is documentation-only and the
omitted checks cannot exercise it.

## Runtime paths and environment

- Global state database: `${DOMYN_SWARM_HOME:-~/.domyn_swarm}/swarm.db`.
- Per-swarm directory:
  `${DOMYN_SWARM_HOME:-~/.domyn_swarm}/swarms/<swarm-name>/`.
- Per-swarm health database: `.../swarms/<swarm-name>/watchdog.db`.
- Load-balancer configuration and logs live in the per-swarm `serving/` and
  `logs/` directories.
- Defaults search order is `DOMYN_SWARM_DEFAULTS`, `./defaults.yaml`,
  `./.domyn_swarm/defaults.yaml`, then `~/.domyn_swarm/defaults.yaml`; the first
  existing file wins.
- `.env` is loaded from the current repository directory and
  `~/.domyn_swarm/.env`.
- User-facing settings include `DOMYN_SWARM_HOME`, `DOMYN_SWARM_LOG_LEVEL`,
  `DOMYN_SWARM_DEFAULTS`, `DOMYN_SWARM_SKIP_DB_UPGRADE`, and
  `DOMYN_SWARM_ASCII`.
- Platform tokens include `DOMYN_SWARM_API_TOKEN`, `VLLM_API_KEY`,
  `SINGULARITYENV_VLLM_API_KEY`, `LEPTONAI_API_TOKEN`, and
  `LEPTON_WORKSPACE_ID`.
