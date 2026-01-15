# Repo Review Recap and Tracking

This document captures the original review feedback and tracks what remains.

## Scope
- README and core orchestration: `src/domyn_swarm/core`, `src/domyn_swarm/config`, `src/domyn_swarm/deploy`, `src/domyn_swarm/backends`
- Jobs / checkpoint / runtime
- State
- CLI / TUI / helpers

## Status Summary
- Jobs / Checkpoint / Runtime: mostly addressed, remaining cleanup
- Runtime plumbing: complete (watchdog args builder + status helper wired)
- Config / Plan / Deployment: complete
- Core Swarm / State: not started
- Backends (Serving + Compute): not started
- CLI / TUI / Helpers: partially addressed (status view updates)

## Planning & Deployment
- [x] Centralize plan assembly (PlanBuilder): normalize resources/env/images across backends.
- [x] Split validation from planning: move `_plan` generation out of `DomynLLMSwarmConfig` validators.
- [x] Add DeploymentContext with serving_spec/extras/timeout/shared env; make `Deployment.up` use it.

## Swarm Lifecycle & State
- [ ] Extract lifecycle helpers (SwarmLifecycle, JobSubmitter) and shrink `DomynLLMSwarm`.
- [ ] Rehydrate state via factory; avoid private `_deployment` mutation.
- [ ] Introduce `SwarmPaths` value object and reuse across swarm/jobs/checkpoints/logs/watchdog DB.

## Backends (Serving + Compute)
- [ ] Implement real `wait` and `cancel` for Slurm compute; standardize JobHandle meta schema.
- [ ] Replace `SrunCommandBuilder` flag munging with a structured `SrunResources` model + tests.
- [ ] Unify readiness probing (scheduler + HTTP + watchdog DB) with a reusable probe abstraction.
- [ ] Extract Lepton spec builders (mount/env/token) for testability and extensibility.

## Jobs Pipeline
- [x] Single execution path: `transform_items` + `run_job_unified`; legacy paths removed.
- [ ] Remove legacy API surface completely (`SwarmJob.transform` and deprecated deps).
- [x] BatchExecutor worker loop refactor + progress hooks + tests.
- [ ] Track per-batch progress independently of tqdm when `progress=False`.
- [ ] Handle `CancelledError` in workers and optionally expose an `on_cancel` hook.
- [ ] Expose `SwarmJob.run` progress toggle (optional) to allow quiet runs while keeping hooks.

## Checkpointing
- [x] Atomic writes, parts, fingerprint validation, tests.
- [ ] Cross-process locking strategy if required (NFS-safe locking).
- [ ] Consider hydrating `done_df` from parts for long-running jobs that never call `finalize()`,
      so partial results are inspectable.
- [ ] Add `fingerprint_mode` (full vs sample) or stable `repr` hashing to reduce overhead on
      large datasets.

## Runtime Observability
- [x] Watchdog defaults aligned and collector payload validation.
- [x] Watchdog args builder + status helper; wired into Slurm template and CLI/TUI.
- [ ] Add `busy_timeout` or reuse SQLAlchemy session to avoid SQLite lock errors when the
      collector is writing.
- [ ] Add a Jinja render test for `build_watchdog_args` usage in Slurm templates.

## Data Backend (pandas / polars / ray)
- [ ] Define a common backend interface (read, slice/shard, write, schema).
- [ ] Normalize conversions between pandas/polars/ray datasets.
- [ ] Align checkpointing across backends (consistent IDs, schema, and fingerprinting).
- [ ] Add tests for each backend and mixed conversions (pandas → polars → pandas).

## CLI / TUI / Helpers
- [ ] Extract SwarmService operations from Typer commands (thin CLI).
- [ ] Consolidate TUI output formatting (table/JSON/YAML) behind a view layer.
- [ ] Improve logging setup with a module-level `get_logger` and handler reuse.
- [x] Surface per-replica rows in status view (CLI/TUI).

## Developer Experience
- [ ] Add a `progress` flag to `SwarmJob.run` and expose progress hooks for programmatic callers.
- [ ] Provide a stable data-backend interface (pandas/polars/ray) to reduce manual conversions.
- [ ] Add a status snapshot API (serving + watchdog rows) for SDK consumers.
- [ ] Add CLI `--json/--yaml` output modes for `status` and `swarm list`.
- [ ] Add `--no-progress` to job submission to silence CLI progress output.
- [ ] Add a `swarm logs` command to tail replica logs + watchdog status.
- [ ] Expand troubleshooting docs (checkpoints, watchdog DB, common Slurm failures).

## SwarmJob API Proposals (Dev Experience)
These are additive proposals; they should not change existing behavior and can be introduced
gradually with compatibility shims.
- [ ] Add `transform_item(item)` (single-item) and let the framework batch it.
  - Example:
    ```python
    class MyJob(SwarmJob):
        output_cols = ["result"]
        async def transform_item(self, item: str) -> str:
            resp = await self.client.completions.create(model=self.model, prompt=item)
            return resp.choices[0].text
    ```
- [ ] Allow dict outputs with an `output_schema` and avoid tuple indexing.
  - Example:
    ```python
    class MyJob(SwarmJob):
        output_schema = ["completion", "score"]
        async def transform_items(self, items: list[str]) -> list[dict]:
            return [{"completion": "...", "score": 0.9} for _ in items]
    ```
- [ ] Provide a `@job` decorator / factory that fills defaults and validates `output_schema`.
  - Example:
    ```python
    @job(output_schema=["result"])
    async def my_job(item: str, client, model, **kwargs) -> dict:
        resp = await client.completions.create(model=model, prompt=item)
        return {"result": resp.choices[0].text}
    ```
- [ ] Add `SwarmJob.run(progress=False, checkpoint=False)` flags for local dev and notebooks.

## Step-by-Step Improvement Plan
1) Stabilize planning: PlanBuilder + tests for slurm/lepton plan outputs.
2) Refactor swarm lifecycle: lifecycle helpers + state rehydrate factory.
3) Harden compute backends: real wait/cancel and job handle meta schema.
4) Complete job pipeline cleanup: remove legacy APIs, finalize checkpoint locking strategy.
5) Align runtime config: template rendering tests and watchdog DB helper usage expansion.
6) Thin CLI / TUI: service layer + consolidated view models and outputs.
