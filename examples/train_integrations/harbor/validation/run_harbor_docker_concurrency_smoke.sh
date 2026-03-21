#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"

JOB_ID="${JOB_ID:-}"
HEAD_NODE="${HEAD_NODE:-}"
DOCKER_MODE="${DOCKER_MODE:-rootful}"
TRIAL_COUNT="${TRIAL_COUNT:-64}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-64}"
MAX_FAILURES="${MAX_FAILURES:-0}"
DISABLE_PROJECT_NETWORK="${DISABLE_PROJECT_NETWORK:-false}"
OUTPUT_ROOT_DEFAULT="$(cd "$REPO_ROOT/.." && pwd)/tmp_logs/harbor-docker-concurrency-smoke-job${JOB_ID}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$OUTPUT_ROOT_DEFAULT}"

if [ -z "$JOB_ID" ]; then
  echo "Missing required environment variable: JOB_ID" >&2
  exit 1
fi

if [ -z "$HEAD_NODE" ]; then
  echo "Missing required environment variable: HEAD_NODE" >&2
  exit 1
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python env not found: $PYTHON_BIN" >&2
  exit 1
fi

DOCKER_HOST_DEFAULT="unix:///tmp/xdg-test-$USER/docker.sock"
if [ "$DOCKER_MODE" = rootful ]; then
  DOCKER_HOST_DEFAULT="unix:///var/run/docker.sock"
fi
DOCKER_HOST="${DOCKER_HOST:-$DOCKER_HOST_DEFAULT}"

disable_project_network_arg=()
if [ "$DISABLE_PROJECT_NETWORK" = true ]; then
  disable_project_network_arg+=(--disable-project-network)
fi

srun --jobid "$JOB_ID" --overlap --overcommit --immediate=10 -w "$HEAD_NODE" \
  --ntasks=1 --nodes=1 --cpus-per-task=4 --gres=gpu:0 \
  bash -lc "cd '$REPO_ROOT' && export DOCKER_HOST='$DOCKER_HOST' && '$PYTHON_BIN' '$SCRIPT_DIR/harbor_docker_concurrency_smoke.py' --output-root '$OUTPUT_ROOT' --trial-count '$TRIAL_COUNT' --max-concurrency '$MAX_CONCURRENCY' --max-failures '$MAX_FAILURES' --docker-host '$DOCKER_HOST' --clean ${disable_project_network_arg[*]}"
