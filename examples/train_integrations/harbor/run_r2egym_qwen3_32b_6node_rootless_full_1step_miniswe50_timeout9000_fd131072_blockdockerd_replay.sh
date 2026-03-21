#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"

cd "$REPO_ROOT"
export PATH="$REPO_ROOT/.venv/bin:$HOME/.local/bin:$PATH"

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_SHORT_ID="${RUN_SHORT_ID:-r2e1srp-$(date +%m%d%H%M%S)}"
RUN_NAME_OVERRIDE="${RUN_NAME_OVERRIDE:-r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-blockdockerd-replay-${RUN_TS}}"

export JOB_ID="${JOB_ID:-${SLURM_JOB_ID:-}}"
export HEAD_NODE="${HEAD_NODE:-research-dev-coder-003}"
export ROLLOUT_NODE="${ROLLOUT_NODE:-research-dev-coder-008}"
export TRAINER_NODES_CSV="${TRAINER_NODES_CSV:-research-dev-coder-012,research-dev-coder-013,research-dev-coder-014,research-dev-coder-015}"

export DOCKER_MODE=rootless
export RAY_PORT="${RAY_PORT:-6381}"
export ROLLOUT_SERVER_PORTS_CSV="${ROLLOUT_SERVER_PORTS_CSV:-18000,18001,18002,18003}"
export ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-2}"
export ROLLOUT_ENGINES="${ROLLOUT_ENGINES:-4}"

export HEAD_NOFILE_SOFT="${HEAD_NOFILE_SOFT:-131072}"
export TRAINER_NOFILE_SOFT="${TRAINER_NOFILE_SOFT:-131072}"
export ROLLOUT_NOFILE_SOFT="${ROLLOUT_NOFILE_SOFT:-131072}"
export DOCKER_NOFILE_SOFT="${DOCKER_NOFILE_SOFT:-131072}"
export DOCKER_INOTIFY_MAX_USER_INSTANCES="${DOCKER_INOTIFY_MAX_USER_INSTANCES:-8192}"
export HEAD_DOCKER_READY_TIMEOUT_SEC="${HEAD_DOCKER_READY_TIMEOUT_SEC:-300}"

export AGENT_TIMEOUT_SEC="${AGENT_TIMEOUT_SEC:-9000}"
export MINI_SWE_MODEL_TIMEOUT_SEC="${MINI_SWE_MODEL_TIMEOUT_SEC:-1200}"
export HARBOR_AGENT_MAX_TURNS="${HARBOR_AGENT_MAX_TURNS:-20}"
export MAX_TRAIN_TASKS="${MAX_TRAIN_TASKS:-64}"
export MONITOR_INTERVAL_SEC="${MONITOR_INTERVAL_SEC:-10}"

export TRAIN_DATA="${TRAIN_DATA:-['/home/hkang/zthunder_agent/data/harbor/r2egym-trivial','/home/hkang/zthunder_agent/data/harbor/r2egym-easy','/home/hkang/zthunder_agent/data/harbor/r2egym-medium','/home/hkang/zthunder_agent/data/harbor/r2egym-hard']}"
export EVAL_DATA="${EVAL_DATA:-$TRAIN_DATA}"

export LOG_DIR="${LOG_DIR:-$WORKSPACE_ROOT/tmp_logs/$RUN_NAME_OVERRIDE}"
export RUN_ARTIFACT_ROOT="${RUN_ARTIFACT_ROOT:-$WORKSPACE_ROOT}"

export ROOTLESS_SCRATCH_ROOT="${ROOTLESS_SCRATCH_ROOT:-/tmp/$USER/${RUN_SHORT_ID}-rootless-scratch}"
export TRAIN_RUNTIME_SCRATCH_ROOT="${TRAIN_RUNTIME_SCRATCH_ROOT:-/tmp/$USER/${RUN_SHORT_ID}-train-runtime}"
export ROLLOUT_RUNTIME_SCRATCH_ROOT="${ROLLOUT_RUNTIME_SCRATCH_ROOT:-/tmp/$USER/${RUN_SHORT_ID}-rollout-runtime}"

DEFAULT_XDG_RUNTIME_DIR="/tmp/xdg-${RUN_SHORT_ID}"
case "${XDG_RUNTIME_DIR:-}" in
  ""|/run/user/*)
    export XDG_RUNTIME_DIR="$DEFAULT_XDG_RUNTIME_DIR"
    ;;
  *)
    export XDG_RUNTIME_DIR
    ;;
esac

case "${DOCKER_HOST:-}" in
  ""|unix:///run/user/*)
    export DOCKER_HOST="unix://$XDG_RUNTIME_DIR/docker.sock"
    ;;
  *)
    export DOCKER_HOST
    ;;
esac

case "${DOCKER_PIDFILE:-}" in
  ""|/run/user/*)
    export DOCKER_PIDFILE="$XDG_RUNTIME_DIR/docker.pid"
    ;;
  *)
    export DOCKER_PIDFILE
    ;;
esac

export DOCKER_EXEC_ROOT="${DOCKER_EXEC_ROOT:-/tmp/$USER/${RUN_SHORT_ID}-rootless-exec}"
export DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT:-/scratch/triton_cache/$USER/r2egym-rootless-image-cache}"
export DOCKER_LOG_PATH="${DOCKER_LOG_PATH:-$LOG_DIR/dockerd_rootless_harbor.log}"

exec bash "$SCRIPT_DIR/run_r2egym_qwen3_32b_6node_rootless_full_1step_miniswe50_timeout9000.sh"
