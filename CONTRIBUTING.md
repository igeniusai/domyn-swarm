# Contributing to Domyn Swarm

Contributions should keep Domyn Swarm predictable for users and maintainable for
contributors. Keep each pull request focused and explain the behavior it changes
or preserves.

## Principles

- Start from the user-facing behavior and make invalid states difficult to
  express or submit.
- Preserve existing behavior and compatibility unless the change explicitly
  introduces a migration.
- Prefer direct designs, existing abstractions, and domain-specific names.
  Generalize code only when current callers share a real rule.
- Keep code, tests, documentation, and configuration descriptions consistent.
- Confirm that copied or adapted code is compatible with the Apache-2.0 license,
  retain required notices, and credit its source.

## Development setup

Domyn Swarm supports Python `>=3.10,<3.14` on Linux. Install
[uv](https://docs.astral.sh/uv/), clone the repository, and synchronize the
development environment:

```bash
uv sync --all-extras
```

Documentation work also needs the docs dependency group:

```bash
uv sync --all-extras --group docs
```

Run project tools through `uv run`. When dependencies change intentionally,
update and commit `uv.lock`; otherwise leave it unchanged.

## Code and prose style

Ruff owns formatting and linting, and Pyright owns static type checking. The
repository-specific conventions are:

- Preserve public imports, serialized forms, CLI output, templates, and
  persistence formats unless the task explicitly changes them.
- Add Google-style docstrings to supported public APIs and non-obvious internal
  contracts. Do not add boilerplate docstrings to obvious private helpers,
  overrides, callbacks, validators, properties, or special methods.
- Comments are exceptional. Use one only when unusual code, a hidden invariant,
  an external constraint, a non-local side effect, or a non-obvious tradeoff
  would otherwise surprise a careful reader. Do not narrate straightforward
  code.
- Use concise active voice, US English, and sentence-case headings. Use “Domyn
  Swarm” for the project, `domyn-swarm` for the CLI and distribution, and
  `domyn_swarm` for the Python package.

Run the style checks with:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pyright
```

## Tests and documentation

Run the closest tests while iterating, then the full suite before review:

```bash
uv run pytest tests/path/to/test_file.py -q --no-cov
uv run pytest
```

Changes to user-facing behavior require corresponding documentation. Build the
documentation with warnings treated as errors:

```bash
docs_output=$(mktemp -d)
DOCS_VERSION=latest uv run sphinx-build -W --keep-going -b html docs "$docs_output"
```

Run every configured hook before submitting:

```bash
uv run pre-commit run --all-files
```

### End-to-end tests

The Slurm end-to-end tests start a three-node Slurm cluster in Docker Compose
and run the `domyn-swarm` CLI against it. The `ctl` container is the Slurm
controller and the submit host. The `c1` and `c2` containers are compute nodes.

The cluster has no GPUs, no Singularity, and no environment modules. Stand-in
executables in `tests/e2e/slurm/shims/` replace `singularity` and `module`.
Each compute node declares two fake GPUs to Slurm. The suite has two tiers:

| Tier | Model server | Scenarios | Image size | CI run time |
| --- | --- | --- | --- | --- |
| `fast` | A small fake OpenAI server | Lifecycle, replica crash and restart | 3.1 GB | About 4 minutes |
| `vllm` | CPU vLLM with `hmellor/tiny-random-LlamaForCausalLM` | Lifecycle | 6 GB | About 6 minutes |

The run times assume a cached image. The first build of the `vllm` image
takes about 10 more minutes.

The tests need Docker Engine with Compose v2 on an amd64 or arm64 host. For
the `vllm` tier, an x86_64 host also needs AVX2. Run a tier from the
repository root:

```bash
tests/e2e/slurm/run.sh fast
tests/e2e/slurm/run.sh vllm
```

Arguments after the tier replace the default pytest target. Test paths are
container paths:

```bash
tests/e2e/slurm/run.sh fast /workspace/tests/e2e/slurm/scenarios/test_lifecycle.py
```

Most changes do not need these tests. Run the `fast` tier before review if you
change one of these areas:

- The Slurm serving or compute backends in `src/domyn_swarm/backends/`.
- The Slurm templates in `src/domyn_swarm/templates/`.
- The watchdog, the collector, or the load-balancer supervisor in
  `src/domyn_swarm/runtime/`.
- The `up`, `status`, or `down` commands.

Run the `vllm` tier if you change the `vllm serve` command line in the
templates or the readiness checks.

The `E2E (Slurm)` workflow runs the `fast` tier on pull requests that change
these areas, on pushes to `main`, and every night. It runs the `vllm` tier
every night. You can start either tier from the Actions tab.

If a run fails, `run.sh` copies the swarm logs and the Slurm daemon logs to
`output/e2e-logs/`. In CI, the workflow uploads this directory as an artifact.

The shim makes each Singularity bind mount a symlink, so a compute node can
host only one swarm at a time, and the scenarios run one after another. The
compute containers run with `CAP_SYS_ADMIN` and without an AppArmor profile.
slurmd needs both to load a login environment for the load-balancer job, as
it does as root on a real node.

The `vllm` image has no C++ compiler, so the `vllm` tier starts vLLM with
`-O0` and sends greedy requests. Both choices keep vLLM away from
`torch.compile`.

The suite does not cover real Singularity, real GPUs, Ray replicas, or the
monitoring sidecars. Without Docker, `uv run pytest` collects the scenarios
and reports them as skipped.

## Pull request guidelines

Open pull requests against `main` and keep each one centered on a single change.
Describe the motivation, observable effect, verification performed, and any
compatibility or migration considerations. Update tests and documentation where
the behavior requires them.

Use conventional commit messages. Every commit must be signed and signed off;
create commits with:

```bash
git commit -sS -m "type(scope): concise description"
```

See GitHub's documentation for
[configuring commit signing](https://docs.github.com/en/authentication/managing-commit-signature-verification/signing-commits).

## Developer Certificate of Origin

Signing off a commit certifies the following Developer Certificate of Origin.

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

(b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

(c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

(d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
```
