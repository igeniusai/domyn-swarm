#!/usr/bin/env bash
# Run the Slurm end-to-end tests in a three-node Docker Compose cluster.
#
# Usage: tests/e2e/slurm/run.sh fast|vllm [pytest args...]
# Pytest arguments replace the default target, so give a container path with them.
# Set E2E_SKIP_BUILD=1 to use an image that is already built (CI builds it with a cache).
set -euo pipefail

tier=${1:?usage: run.sh fast|vllm [pytest args...]}
shift
case "$tier" in
  fast) export E2E_TIMEOUT_S=${E2E_TIMEOUT_S:-600} ;;
  vllm) export E2E_TIMEOUT_S=${E2E_TIMEOUT_S:-1800} ;;
  *) echo "unknown tier: $tier (expected fast or vllm)" >&2; exit 2 ;;
esac
export E2E_TARGET=$tier

here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
logs="$repo/output/e2e-logs"
compose=(docker compose -f "$here/compose.yaml")

cleanup() { "${compose[@]}" down -v --remove-orphans >/dev/null 2>&1 || true; }
trap cleanup EXIT

if [ "${E2E_SKIP_BUILD:-0}" != 1 ]; then
  "${compose[@]}" build ctl
fi
"${compose[@]}" up -d --no-build

echo "waiting for c1 and c2 to become idle"
idle=0
for _ in $(seq 1 60); do
  idle=$("${compose[@]}" exec -T ctl sinfo -h -N -t idle -o %N 2>/dev/null | sort -u | wc -l || true)
  [ "$idle" -eq 2 ] && break
  sleep 2
done
if [ "$idle" -ne 2 ]; then
  echo "the Slurm compute nodes did not become idle" >&2
  "${compose[@]}" exec -T ctl sinfo -N -l >&2 || true
  "${compose[@]}" exec -T ctl cat /shared/logs/slurmctld.log /shared/logs/slurmd-c1.log >&2 || true
  exit 1
fi

"${compose[@]}" exec -T ctl uv sync --frozen --project /workspace

if [ $# -eq 0 ]; then
  set -- /workspace/tests/e2e/slurm/scenarios
fi
status=0
# Jobs inherit the working directory; /workspace is read-only, /shared exists on every node.
"${compose[@]}" exec -T -w /shared ctl /shared/venv/bin/python -m pytest \
  -p no:cacheprovider --no-cov -v "$@" || status=$?

if [ "$status" -ne 0 ]; then
  rm -rf "$logs"
  mkdir -p "$logs"
  "${compose[@]}" cp ctl:/shared/logs "$logs/slurm" || true
  "${compose[@]}" cp ctl:/shared/home/swarms "$logs/swarms" || true
  echo "logs copied to $logs" >&2
fi
exit "$status"
