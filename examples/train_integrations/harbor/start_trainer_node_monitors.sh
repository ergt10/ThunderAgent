#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"

JOB_ID="${JOB_ID:-}"
TRAINER_NODES_CSV="${TRAINER_NODES_CSV:-}"
RUN_NAME="${RUN_NAME:-}"
MONITOR_ROOT="${MONITOR_ROOT:-$WORKSPACE_ROOT/tmp_logs/$RUN_NAME/trainer_monitors}"
TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-$WORKSPACE_ROOT/tmp_logs/$RUN_NAME}"
MONITOR_INTERVAL_SEC="${MONITOR_INTERVAL_SEC:-10}"
SRUN_CPUS_PER_TASK="${SRUN_CPUS_PER_TASK:-1}"

require_env() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "Missing required environment variable: $name" >&2
    exit 1
  fi
}

require_env JOB_ID
require_env TRAINER_NODES_CSV
require_env RUN_NAME

IFS=',' read -r -a TRAINER_NODES <<<"$TRAINER_NODES_CSV"
if [ "${#TRAINER_NODES[@]}" -eq 0 ]; then
  echo "TRAINER_NODES_CSV is empty" >&2
  exit 1
fi

mkdir -p "$MONITOR_ROOT"

SRUN_PIDS=()

cleanup() {
  if [ "${#SRUN_PIDS[@]}" -eq 0 ]; then
    return 0
  fi
  kill "${SRUN_PIDS[@]}" 2>/dev/null || true
  wait "${SRUN_PIDS[@]}" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

for node in "${TRAINER_NODES[@]}"; do
  node_output_dir="$MONITOR_ROOT/$node"
  node_tb_dir="$node_output_dir/tensorboard"
  launcher_log="$MONITOR_ROOT/${node}.launcher.log"
  mkdir -p "$node_output_dir" "$node_tb_dir"

  srun --jobid "$JOB_ID" --overlap --overcommit --exact \
    -w "$node" --ntasks=1 --nodes=1 --cpus-per-task="$SRUN_CPUS_PER_TASK" --gres=gpu:0 \
    bash -lc "cd '$REPO_ROOT' && export TENSORBOARD_DIR='$node_tb_dir' && bash '$SCRIPT_DIR/monitor_stage3_resources.sh' '$RUN_NAME-trainer-$node' '$MONITOR_INTERVAL_SEC' '$node_output_dir' --tensorboard-dir '$node_tb_dir' --log-dir '$TRAIN_LOG_DIR'" \
    >"$launcher_log" 2>&1 &
  SRUN_PIDS+=("$!")
  echo "Started trainer monitor on $node"
  echo "  output_dir: $node_output_dir"
  echo "  launcher_log: $launcher_log"
done

wait "${SRUN_PIDS[@]}"
