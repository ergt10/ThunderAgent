#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
RAY_BIN="${RAY_BIN:-$REPO_ROOT/.venv/bin/ray}"

JOB_ID="${JOB_ID:-${SLURM_JOB_ID:-}}"
HEAD_NODE="${HEAD_NODE:-research-dev-coder-003}"
ROLLOUT_NODE="${ROLLOUT_NODE:-research-dev-coder-008}"
TRAINER_NODES_CSV="${TRAINER_NODES_CSV:-research-dev-coder-012,research-dev-coder-013,research-dev-coder-014,research-dev-coder-015}"
DOCKER_MODE="${DOCKER_MODE:-rootful}"
HARBOR_DOCKER_DISABLE_PROJECT_NETWORK="${HARBOR_DOCKER_DISABLE_PROJECT_NETWORK:-0}"
RAY_PORT="${RAY_PORT:-6381}"
ROLLOUT_PORT_A="${ROLLOUT_PORT_A:-18000}"
ROLLOUT_PORT_B="${ROLLOUT_PORT_B:-18001}"
ROLLOUT_SERVER_PORTS_CSV="${ROLLOUT_SERVER_PORTS_CSV:-${ROLLOUT_PORT_A},${ROLLOUT_PORT_B}}"
ROLLOUT_GPU_GROUPS_SPEC="${ROLLOUT_GPU_GROUPS_SPEC:-}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-4}"
MONITOR_INTERVAL_SEC="${MONITOR_INTERVAL_SEC:-10}"
TARGET_STEPS="${TARGET_STEPS:-10}"
RUN_ROLLOUT_CUDA_SMOKE="${RUN_ROLLOUT_CUDA_SMOKE:-true}"
FULL_POLICY_MINI_BATCH_SIZE="${FULL_POLICY_MINI_BATCH_SIZE:-64}"
MAX_TRAIN_TASKS="${MAX_TRAIN_TASKS:-$((FULL_POLICY_MINI_BATCH_SIZE * TARGET_STEPS))}"
RUN_NAME="${RUN_NAME_OVERRIDE:-codecontest-qwen3-32b-6node-full-${TARGET_STEPS}step-$(date +%Y%m%d_%H%M%S)}"
ROOTLESS_XDG_RUNTIME_DIR="${ROOTLESS_XDG_RUNTIME_DIR:-/tmp/xdg-test-$USER}"
if [ "$DOCKER_MODE" = rootful ]; then
  XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-$ROOTLESS_XDG_RUNTIME_DIR}"
  DOCKER_HOST="${DOCKER_HOST:-unix:///var/run/docker.sock}"
else
  XDG_RUNTIME_DIR="$ROOTLESS_XDG_RUNTIME_DIR"
  DOCKER_HOST="${DOCKER_HOST:-unix://$XDG_RUNTIME_DIR/docker.sock}"
fi
HEAD_NOFILE_SOFT="${HEAD_NOFILE_SOFT:-131072}"
LOG_ROOT="${LOG_ROOT:-$WORKSPACE_ROOT/tmp_logs}"
LOG_DIR="${LOG_DIR_OVERRIDE:-$LOG_ROOT/$RUN_NAME}"
RUN_ARTIFACT_ROOT="${RUN_ARTIFACT_ROOT:-$WORKSPACE_ROOT/harbor_runs}"
MIN_HOME_RUN_FREE_GB="${MIN_HOME_RUN_FREE_GB:-500}"
SKIP_ROOTLESS_HARBOR_CHECK="${SKIP_ROOTLESS_HARBOR_CHECK:-true}"

TRAINER_MONITOR_ROOT="$LOG_DIR/trainer_monitors"
ROLLOUT_LOG_DIR="$LOG_DIR/rollout"
ROLLOUT_MONITOR_DIR="$ROLLOUT_LOG_DIR/monitoring"
TRAIN_TENSORBOARD_DIR="$LOG_DIR/tensorboard"
SUMMARY_DIR="$LOG_DIR/post_run_summary"
RAY_LOG="$LOG_DIR/launcher_ray.log"
ROLLOUT_LOG="$LOG_DIR/launcher_rollout.log"
TRAINER_MONITOR_LOG="$LOG_DIR/launcher_trainer_monitors.log"
VALIDATION_LOG="$LOG_DIR/launcher_validation.log"
TRAIN_DRIVER_LOG="$LOG_DIR/launcher_train_driver.log"
SUMMARY_LOG="$LOG_DIR/launcher_summary.log"
ROLLOUT_CUDA_SMOKE_LOG="$LOG_DIR/launcher_rollout_cuda_smoke.log"
ROOTLESS_DOCKER_LOG="$LOG_DIR/launcher_rootless_docker.log"

require_env() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "Missing required environment variable: $name" >&2
    exit 1
  fi
}

require_cmd() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || {
    echo "Missing required command: $cmd" >&2
    exit 1
  }
}

detect_job_id() {
  if [ -n "$JOB_ID" ]; then
    return 0
  fi
  mapfile -t running_jobs < <(squeue -u "$USER" -h -t R -o '%i')
  if [ "${#running_jobs[@]}" -eq 1 ]; then
    JOB_ID="${running_jobs[0]}"
    return 0
  fi
  return 1
}

node_ip() {
  local node="$1"
  srun --jobid "$JOB_ID" --overlap --overcommit -w "$node" --ntasks=1 --nodes=1 --cpus-per-task=1 --gres=gpu:0 \
    bash -lc "hostname -I | awk '{print \$1}'"
}

trim_csv_array() {
  local -n ref="$1"
  local idx=""
  for idx in "${!ref[@]}"; do
    ref[$idx]="$(printf '%s' "${ref[$idx]}" | xargs)"
  done
}

server_source_name() {
  local idx="$1"
  local suffixes=(a b c d e f g h i j k l m n o p)
  if [ "$idx" -lt "${#suffixes[@]}" ]; then
    printf 'rollout_%s' "${suffixes[$idx]}"
    return
  fi
  printf 'rollout_%02d' "$idx"
}

build_rollout_server_urls_literal() {
  local host_ip="$1"
  shift
  local -a urls=()
  local port=""
  for port in "$@"; do
    urls+=("\"http://${host_ip}:${port}\"")
  done
  printf '[%s]\n' "$(IFS=,; echo "${urls[*]}")"
}

build_rollout_monitor_specs() {
  local host_ip="$1"
  shift
  local -a specs=()
  local ports=("$@")
  local idx=""
  for idx in "${!ports[@]}"; do
    specs+=("$(server_source_name "$idx").log=http://${host_ip}:${ports[$idx]}")
  done
  printf '%s\n' "$(IFS=';'; echo "${specs[*]}")"
}

wait_for_ray() {
  local ray_address="$1"
  local timeout_sec="${2:-180}"
  local deadline=$((SECONDS + timeout_sec))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if "$RAY_BIN" status --address "$ray_address" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "Timed out waiting for Ray at $ray_address" >&2
  return 1
}

wait_for_rollout_health() {
  local host="$1"
  local port="$2"
  local name="$3"
  local timeout_sec="${4:-900}"
  local deadline=$((SECONDS + timeout_sec))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if curl -sf "http://$host:$port/health" >/dev/null; then
      return 0
    fi
    sleep 2
  done
  echo "Timed out waiting for $name at http://$host:$port/health" >&2
  return 1
}

wait_for_head_docker() {
  local node="$1"
  local docker_host="$2"
  local timeout_sec="${3:-120}"
  local deadline=$((SECONDS + timeout_sec))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if srun --jobid "$JOB_ID" --overlap --overcommit -w "$node" --ntasks=1 --nodes=1 --cpus-per-task=1 --gres=gpu:0 \
      bash -lc "DOCKER_HOST='$docker_host' timeout 10 docker info >/dev/null 2>&1"; then
      return 0
    fi
    sleep 2
  done
  echo "Timed out waiting for Docker at $docker_host on $node" >&2
  return 1
}

check_home_headroom() {
  local target="$1"
  local min_gb="$2"
  local available_gb
  mkdir -p "$target"
  available_gb="$(df -BG "$target" | awk 'NR==2 { gsub(/G/, "", $4); print $4 }')"
  echo "/home artifact root available_gb=$available_gb min_gb=$min_gb"
  [ -n "$available_gb" ] || {
    echo "Failed to read free space for $target" >&2
    exit 1
  }
  [ "$available_gb" -ge "$min_gb" ] || {
    echo "Insufficient free space at $target" >&2
    exit 1
  }
}

PIDS=()

cleanup() {
  if [ "${#PIDS[@]}" -eq 0 ]; then
    return 0
  fi
  kill "${PIDS[@]}" 2>/dev/null || true
  wait "${PIDS[@]}" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

require_cmd srun
require_cmd squeue
require_cmd curl
detect_job_id || require_env JOB_ID

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python env not found: $PYTHON_BIN" >&2
  exit 1
fi
if [ ! -x "$RAY_BIN" ]; then
  echo "Ray CLI not found: $RAY_BIN" >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "$TRAINER_MONITOR_ROOT" "$SUMMARY_DIR" "$RUN_ARTIFACT_ROOT"
check_home_headroom "$RUN_ARTIFACT_ROOT" "$MIN_HOME_RUN_FREE_GB"

IFS=',' read -r -a ROLLOUT_PORTS <<<"$ROLLOUT_SERVER_PORTS_CSV"
trim_csv_array ROLLOUT_PORTS
ROLLOUT_ENGINES="${ROLLOUT_ENGINES:-${#ROLLOUT_PORTS[@]}}"

HEAD_IP="$(node_ip "$HEAD_NODE")"
ROLLOUT_IP="$(node_ip "$ROLLOUT_NODE")"
ROLLOUT_SERVER_URLS="$(build_rollout_server_urls_literal "$ROLLOUT_IP" "${ROLLOUT_PORTS[@]}")"
ROLLOUT_METRICS_ENDPOINT_SPECS="$(build_rollout_monitor_specs "$ROLLOUT_IP" "${ROLLOUT_PORTS[@]}")"

echo "${TARGET_STEPS}-step full-run topology"
echo "  job_id:        $JOB_ID"
echo "  head_node:     $HEAD_NODE ($HEAD_IP)"
echo "  trainer_nodes: $TRAINER_NODES_CSV"
echo "  rollout_node:  $ROLLOUT_NODE ($ROLLOUT_IP)"
echo "  run_name:      $RUN_NAME"
echo "  log_dir:       $LOG_DIR"
echo "  artifact_root: $RUN_ARTIFACT_ROOT"
echo "  max_train_tasks: $MAX_TRAIN_TASKS"
echo "  rollout_ports: ${ROLLOUT_SERVER_PORTS_CSV}"
echo "  rollout_tp:    $ROLLOUT_TP_SIZE"

if [ "$DOCKER_MODE" = rootless ]; then
  srun --jobid "$JOB_ID" --overlap --overcommit -w "$HEAD_NODE" --ntasks=1 --nodes=1 --cpus-per-task=4 --gres=gpu:0 \
    bash -lc "cd '$REPO_ROOT' && export ROOTLESS_DOCKER_START_MODE=block && export XDG_RUNTIME_DIR='$XDG_RUNTIME_DIR' && export DOCKER_HOST='$DOCKER_HOST' && export DOCKER_NOFILE_SOFT='$HEAD_NOFILE_SOFT' && bash '$SCRIPT_DIR/start_rootless_docker_for_harbor.sh'" \
    >"$ROOTLESS_DOCKER_LOG" 2>&1 &
  PIDS+=("$!")
  wait_for_head_docker "$HEAD_NODE" "$DOCKER_HOST"
fi

if [ "$RUN_ROLLOUT_CUDA_SMOKE" = true ]; then
  JOB_ID="$JOB_ID" \
  ROLLOUT_NODE="$ROLLOUT_NODE" \
  PYTHON_BIN="$PYTHON_BIN" \
  RUN_NAME="$RUN_NAME-rollout-cuda-smoke" \
  LOG_DIR="$LOG_DIR/rollout_cuda_smoke" \
  bash "$SCRIPT_DIR/check_rollout_cuda_runtime_smoke.sh" >"$ROLLOUT_CUDA_SMOKE_LOG" 2>&1
fi

JOB_ID="$JOB_ID" \
HEAD_NODE="$HEAD_NODE" \
TRAINER_NODES_CSV="$TRAINER_NODES_CSV" \
ROLLOUT_NODE="$ROLLOUT_NODE" \
RAY_START_MODE=block \
RAY_PORT="$RAY_PORT" \
bash "$SCRIPT_DIR/launch_qwen3_32b_ray_cluster.sh" >"$RAY_LOG" 2>&1 &
PIDS+=("$!")

wait_for_ray "$HEAD_IP:$RAY_PORT"

srun --jobid "$JOB_ID" --overlap --overcommit -w "$ROLLOUT_NODE" --ntasks=1 --nodes=1 --cpus-per-task=100 --gres=gpu:8 \
  bash -lc "cd '$REPO_ROOT' && export RAY_HEAD_IP='$HEAD_IP' && export RUN_NAME='$RUN_NAME' && export LOG_DIR='$ROLLOUT_LOG_DIR' && export MONITORING_DIR='$ROLLOUT_MONITOR_DIR' && export TENSORBOARD_DIR='$ROLLOUT_LOG_DIR/tensorboard' && export PORT_A='$ROLLOUT_PORT_A' && export PORT_B='$ROLLOUT_PORT_B' && export ROLLOUT_SERVER_PORTS_CSV='$ROLLOUT_SERVER_PORTS_CSV' && export ROLLOUT_GPU_GROUPS_SPEC='$ROLLOUT_GPU_GROUPS_SPEC' && export TP_SIZE='$ROLLOUT_TP_SIZE' && bash '$SCRIPT_DIR/start_qwen3_32b_external_rollout_servers.sh'" \
  >"$ROLLOUT_LOG" 2>&1 &
PIDS+=("$!")

for idx in "${!ROLLOUT_PORTS[@]}"; do
  wait_for_rollout_health "$ROLLOUT_IP" "${ROLLOUT_PORTS[$idx]}" "$(server_source_name "$idx")"
done

JOB_ID="$JOB_ID" \
VALIDATION_MODE=full \
HEAD_NODE="$HEAD_NODE" \
TRAINER_NODES_CSV="$TRAINER_NODES_CSV" \
ROLLOUT_NODE="$ROLLOUT_NODE" \
ROLLOUT_SERVER_URLS="$ROLLOUT_SERVER_URLS" \
DOCKER_MODE="$DOCKER_MODE" \
XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR" \
DOCKER_HOST="$DOCKER_HOST" \
HEAD_NOFILE_SOFT="$HEAD_NOFILE_SOFT" \
SKIP_ROLLOUT_CUDA_RUNTIME_SMOKE=true \
SKIP_ROLLOUT_PORT_CHECKS=true \
SKIP_ROOTLESS_HARBOR_CHECK="$SKIP_ROOTLESS_HARBOR_CHECK" \
bash "$SCRIPT_DIR/run_full_run_validation_suite.sh" >"$VALIDATION_LOG" 2>&1

JOB_ID="$JOB_ID" \
TRAINER_NODES_CSV="$TRAINER_NODES_CSV" \
RUN_NAME="$RUN_NAME" \
MONITOR_ROOT="$TRAINER_MONITOR_ROOT" \
TRAIN_LOG_DIR="$LOG_DIR" \
MONITOR_INTERVAL_SEC="$MONITOR_INTERVAL_SEC" \
bash "$SCRIPT_DIR/start_trainer_node_monitors.sh" >"$TRAINER_MONITOR_LOG" 2>&1 &
PIDS+=("$!")

srun --jobid "$JOB_ID" --overlap --overcommit -w "$HEAD_NODE" --ntasks=1 --nodes=1 --cpus-per-task=8 --gres=gpu:0 \
  bash -lc "cd '$REPO_ROOT' && export DOCKER_MODE='$DOCKER_MODE' && export XDG_RUNTIME_DIR='$XDG_RUNTIME_DIR' && export DOCKER_HOST='$DOCKER_HOST' && export HEAD_NOFILE_SOFT='$HEAD_NOFILE_SOFT' && export HARBOR_DOCKER_DISABLE_PROJECT_NETWORK='$HARBOR_DOCKER_DISABLE_PROJECT_NETWORK' && export RAY_HEAD_IP='$HEAD_IP' && export ROLLOUT_HOST_IP='$ROLLOUT_IP' && export ROLLOUT_SERVER_URLS='$ROLLOUT_SERVER_URLS' && export ROLLOUT_SERVER_PORTS_CSV='$ROLLOUT_SERVER_PORTS_CSV' && export ROLLOUT_ENGINES='$ROLLOUT_ENGINES' && export ROLLOUT_TP_SIZE='$ROLLOUT_TP_SIZE' && export ROLLOUT_METRICS_ENDPOINT_SPECS='$ROLLOUT_METRICS_ENDPOINT_SPECS' && export RUN_NAME_OVERRIDE='$RUN_NAME' && export LOG_DIR_OVERRIDE='$LOG_DIR' && export RUN_ARTIFACT_ROOT='$RUN_ARTIFACT_ROOT' && export CKPT_INTERVAL='$TARGET_STEPS' && export HF_SAVE_INTERVAL='-1' && bash '$SCRIPT_DIR/run_codecontest_qwen3_32b_6node_rootless_fully_async.sh' full max_train_tasks='$MAX_TRAIN_TASKS' trainer.max_ckpts_to_keep=1 trainer.resume_mode=none trainer.save_final_checkpoint_at_end=false trainer.save_final_hf_model_at_end=false" \
  >"$TRAIN_DRIVER_LOG" 2>&1

"$PYTHON_BIN" "$SCRIPT_DIR/summarize_qwen3_32b_full_run.py" \
  --run-name "$RUN_NAME" \
  --log-dir "$LOG_DIR" \
  --trainer-monitor-root "$TRAINER_MONITOR_ROOT" \
  --rollout-monitor-dir "$ROLLOUT_MONITOR_DIR" \
  --tensorboard-dir "$TRAIN_TENSORBOARD_DIR" \
  --async-trace "$LOG_DIR/monitoring/async_trace.jsonl" \
  --output-dir "$SUMMARY_DIR" \
  --python-bin "$PYTHON_BIN" \
  >"$SUMMARY_LOG" 2>&1

echo "${TARGET_STEPS}-step full run completed"
echo "  train_driver_log: $TRAIN_DRIVER_LOG"
echo "  summary_log:      $SUMMARY_LOG"
echo "  summary_dir:      $SUMMARY_DIR"
