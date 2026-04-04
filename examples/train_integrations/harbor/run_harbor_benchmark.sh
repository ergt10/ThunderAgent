#!/usr/bin/env bash
set -euo pipefail

# Canonical Harbor benchmark wrapper.
# Spec summary:
# - workload: R2EGYM
# - model family: Qwen3-32B
# - topology: merged head/rollout plus 4 trainer nodes across one or more Slurm jobs
# - rollout: 4 external servers, TP=2
# - runtime: Harbor patches and task docker-image backfill managed here

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
DEFAULT_RUNTIME_VENV="/data/zy/models/hkang/run_worktrees/skyrl-r2e-clean-20260321/.venv"
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_RUNTIME_VENV/bin/python}"
RAY_BIN="${RAY_BIN:-$DEFAULT_RUNTIME_VENV/bin/ray}"
PYTHON_BIN_DIR="$(dirname "$PYTHON_BIN")"

MERGED_JOB_ID="${MERGED_JOB_ID:-${JOB_ID:-${SLURM_JOB_ID:-}}}"
MERGED_NODE="${MERGED_NODE:-}"
ROLLOUT_JOB_ID="${ROLLOUT_JOB_ID:-}"
ROLLOUT_NODE="${ROLLOUT_NODE:-}"
TRAINER_NODE_SPECS="${TRAINER_NODE_SPECS:-}"
ACTION="${ACTION:-${1:-all}}"
ACTION_ARG="${ACTION_ARG:-${2:-}}"
DOCKER_MODE_RAW="${DOCKER_MODE:-rootless}"

case "$DOCKER_MODE_RAW" in
  rootless)
    DOCKER_MODE="rootless"
    ;;
  rootful|system)
    DOCKER_MODE="rootful"
    ;;
  *)
    echo "Unsupported DOCKER_MODE: $DOCKER_MODE_RAW (expected rootless, rootful, or system)" >&2
    exit 1
    ;;
esac

RAY_PORT="${RAY_PORT:-6381}"
HEAD_CPUS="${HEAD_CPUS:-8}"
HEAD_GPUS="${HEAD_GPUS:-0}"
TRAINER_CPUS="${TRAINER_CPUS:-176}"
TRAINER_GPUS="${TRAINER_GPUS:-8}"
ROLLOUT_CPUS="${ROLLOUT_CPUS:-100}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-8}"
ROLLOUT_SERVER_PORTS_CSV="${ROLLOUT_SERVER_PORTS_CSV:-18000,18001,18002,18003}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-2}"
ROLLOUT_ENGINES="${ROLLOUT_ENGINES:-4}"
ROLLOUT_GPU_GROUPS_SPEC="${ROLLOUT_GPU_GROUPS_SPEC:-}"
HEAD_NOFILE_SOFT="${HEAD_NOFILE_SOFT:-131072}"
TRAINER_NOFILE_SOFT="${TRAINER_NOFILE_SOFT:-131072}"
ROLLOUT_NOFILE_SOFT="${ROLLOUT_NOFILE_SOFT:-131072}"
DOCKER_NOFILE_SOFT="${DOCKER_NOFILE_SOFT:-131072}"
DOCKER_INOTIFY_MAX_USER_INSTANCES="${DOCKER_INOTIFY_MAX_USER_INSTANCES:-}"
HARBOR_DOCKER_SHARED_NETWORK_NAME="${HARBOR_DOCKER_SHARED_NETWORK_NAME:-}"
HARBOR_DOCKER_SHARED_NETWORK_SUBNET="${HARBOR_DOCKER_SHARED_NETWORK_SUBNET:-}"
HEAD_DOCKER_READY_TIMEOUT_SEC="${HEAD_DOCKER_READY_TIMEOUT_SEC:-300}"
ROLLOUT_HEALTH_TIMEOUT_SEC="${ROLLOUT_HEALTH_TIMEOUT_SEC:-900}"
PREPULL_R2EGYM_IMAGES="${PREPULL_R2EGYM_IMAGES:-true}"
DOCKER_PULL_CONCURRENCY="${DOCKER_PULL_CONCURRENCY:-8}"
PREPULL_RESTART_VERIFY_VISIBILITY="${PREPULL_RESTART_VERIFY_VISIBILITY:-true}"
DOCKER_SHUTDOWN_TIMEOUT_SEC="${DOCKER_SHUTDOWN_TIMEOUT_SEC:-30}"
AUTO_APPLY_HARBOR_RUNTIME_PATCHES="${AUTO_APPLY_HARBOR_RUNTIME_PATCHES:-true}"
AUTO_BACKFILL_R2EGYM_DOCKER_IMAGES="${AUTO_BACKFILL_R2EGYM_DOCKER_IMAGES:-true}"
RUN_PREFLIGHT_CHECKS="${RUN_PREFLIGHT_CHECKS:-true}"
AGENT_RUNTIME_PREFLIGHT="${AGENT_RUNTIME_PREFLIGHT:-true}"
ROOTLESS_SUBID_CHECK="${ROOTLESS_SUBID_CHECK:-true}"
AUTO_RAISE_KERNEL_KEY_QUOTA="${AUTO_RAISE_KERNEL_KEY_QUOTA:-true}"
KERNEL_KEYS_MAXKEYS="${KERNEL_KEYS_MAXKEYS:-20000}"
KERNEL_KEYS_MAXBYTES="${KERNEL_KEYS_MAXBYTES:-25000000}"
RUN_HARBOR_DOCKER_CONCURRENCY_SMOKE_PREPARE="${RUN_HARBOR_DOCKER_CONCURRENCY_SMOKE_PREPARE:-true}"
HARBOR_CONCURRENCY_TRIAL_COUNT="${HARBOR_CONCURRENCY_TRIAL_COUNT:-16}"
HARBOR_CONCURRENCY_MAX_IN_FLIGHT="${HARBOR_CONCURRENCY_MAX_IN_FLIGHT:-16}"
HARBOR_CONCURRENCY_MAX_FAILURES="${HARBOR_CONCURRENCY_MAX_FAILURES:-0}"
HARBOR_CONCURRENCY_DISABLE_PROJECT_NETWORK="${HARBOR_CONCURRENCY_DISABLE_PROJECT_NETWORK:-false}"
AUTO_CLEAN_DIRTY_GPUS="${AUTO_CLEAN_DIRTY_GPUS:-true}"
GPU_CLEANUP_USE_SUDO="${GPU_CLEANUP_USE_SUDO:-true}"
GPU_CLEANUP_TERM_TIMEOUT_SEC="${GPU_CLEANUP_TERM_TIMEOUT_SEC:-10}"
GPU_CLEANUP_KILL_TIMEOUT_SEC="${GPU_CLEANUP_KILL_TIMEOUT_SEC:-5}"

AGENT_TIMEOUT_SEC="${AGENT_TIMEOUT_SEC:-9000}"
MINI_SWE_MODEL_TIMEOUT_SEC="${MINI_SWE_MODEL_TIMEOUT_SEC:-1200}"
HARBOR_AGENT_MAX_TURNS="${HARBOR_AGENT_MAX_TURNS:-20}"
MAX_TRAIN_TASKS="${MAX_TRAIN_TASKS:-64}"
MAX_EVAL_TASKS="${MAX_EVAL_TASKS:-20}"
FULL_EPOCHS="${FULL_EPOCHS:-1}"
EVAL_INTERVAL_STEPS="${EVAL_INTERVAL_STEPS:-50}"
CLEANUP_ON_SUCCESS="${CLEANUP_ON_SUCCESS:-true}"
SKYRL_INFERENCE_ROUTER_PORT="${SKYRL_INFERENCE_ROUTER_PORT:-18080}"
CKPT_INTERVAL="${CKPT_INTERVAL:--1}"
HF_SAVE_INTERVAL="${HF_SAVE_INTERVAL:--1}"
CKPT_ROOT_OVERRIDE="${CKPT_ROOT_OVERRIDE:-}"
EXPORT_ROOT_OVERRIDE="${EXPORT_ROOT_OVERRIDE:-}"
TRAINER_RESUME_MODE="${TRAINER_RESUME_MODE:-none}"
TRAINER_RESUME_PATH="${TRAINER_RESUME_PATH:-}"
MSWEA_API_KEY="${MSWEA_API_KEY:-skyrl-local-noauth}"
ALLOW_UNSAFE_WRAPPER_ACTIONS="${ALLOW_UNSAFE_WRAPPER_ACTIONS:-false}"
DRIVER_TERMINAL_WAIT_POLL_SEC="${DRIVER_TERMINAL_WAIT_POLL_SEC:-30}"
DRIVER_TERMINAL_WAIT_STALE_TIMEOUT_SEC="${DRIVER_TERMINAL_WAIT_STALE_TIMEOUT_SEC:-1800}"

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_SHORT_ID="${RUN_SHORT_ID:-r2e5m-$(date +%m%d%H%M%S)}"
RUN_NAME="${RUN_NAME_OVERRIDE:-r2egym-qwen3-32b-5node-rootless-full-1step-4srv-miniswe50-timeout9000-fd131072-crossjob-merged-${RUN_TS}}"
LOG_DIR="${LOG_DIR:-$WORKSPACE_ROOT/tmp_logs/$RUN_NAME}"
RUN_ARTIFACT_ROOT="${RUN_ARTIFACT_ROOT:-/scratch/triton_cache/$USER/harbor_run_artifacts}"
RUN_ARTIFACT_DIR="$RUN_ARTIFACT_ROOT/$RUN_NAME"
ROLLOUT_LOG_DIR="$LOG_DIR/rollout"
ROLLOUT_MONITOR_DIR="$ROLLOUT_LOG_DIR/monitoring"
ROOTLESS_DOCKER_LOG="$LOG_DIR/launcher_rootless_docker.log"
PREPULL_DOCKER_LOG="$LOG_DIR/prepull_rootless_docker.log"
RAY_LOG="$LOG_DIR/launcher_ray.log"
ROLLOUT_LOG="$LOG_DIR/launcher_rollout.log"
TRAIN_DRIVER_LOG="$LOG_DIR/launcher_train_driver.log"
AGENT_RUNTIME_PREFLIGHT_LOG="$LOG_DIR/agent_runtime_preflight.log"
PREPULL_LOG="$LOG_DIR/launcher_prepull.log"
IMAGE_LIST_PATH="$LOG_DIR/r2egym_images.txt"
MINI_SWE_TOOL_INSTALL_LOG="$LOG_DIR/mini_swe_agent_tool_install.log"
LOCAL_TMP_DIR="${LOCAL_TMP_DIR:-$LOG_DIR/local-tmp}"
STATE_DIR="$LOG_DIR/state"
PREPARE_GPU_LOG_DIR="$LOG_DIR/prepare_gpu_cleanup"

MERGED_SCRATCH_ROOT="${MERGED_SCRATCH_ROOT:-/scratch/triton_cache/$USER}"
ROOTLESS_SCRATCH_ROOT="${ROOTLESS_SCRATCH_ROOT:-$MERGED_SCRATCH_ROOT/${RUN_SHORT_ID}-rootless-scratch}"
TRAIN_RUNTIME_SCRATCH_ROOT="${TRAIN_RUNTIME_SCRATCH_ROOT:-/scratch/$USER/skyrl_runtime/${RUN_SHORT_ID}-train-runtime}"
ROLLOUT_RUNTIME_SCRATCH_ROOT="${ROLLOUT_RUNTIME_SCRATCH_ROOT:-/scratch/$USER/skyrl_runtime/${RUN_SHORT_ID}-rollout-runtime}"
MERGED_TMP_DIR="${MERGED_TMP_DIR:-$MERGED_SCRATCH_ROOT/${RUN_SHORT_ID}-tmp}"
PREPULL_VERIFY_ROOT_BASE="${PREPULL_VERIFY_ROOT_BASE:-$MERGED_SCRATCH_ROOT/rv-${RUN_SHORT_ID}}"
HEAD_RAY_TMP_DIR="${HEAD_RAY_TMP_DIR:-/scratch/$USER/raytmp-head}"
TRAINER_RAY_TMP_DIR_ROOT="${TRAINER_RAY_TMP_DIR_ROOT:-/scratch/$USER/raytmp}"
XDG_RUNTIME_DIR="${WRAPPER_XDG_RUNTIME_DIR:-$MERGED_SCRATCH_ROOT/xdg-${RUN_SHORT_ID}}"
DOCKER_DATA_ROOT="${WRAPPER_DOCKER_DATA_ROOT:-$MERGED_SCRATCH_ROOT/r2egym-rootless-image-cache}"
PREPULL_SCRATCH_ROOT="${PREPULL_SCRATCH_ROOT:-$MERGED_SCRATCH_ROOT/${RUN_SHORT_ID}-rootless-scratch-prepull}"

if [ "$DOCKER_MODE" = "rootful" ]; then
  DOCKER_HOST="${WRAPPER_DOCKER_HOST:-unix:///var/run/docker.sock}"
  DOCKER_PIDFILE="${WRAPPER_DOCKER_PIDFILE:-}"
  DOCKER_EXEC_ROOT="${WRAPPER_DOCKER_EXEC_ROOT:-}"
  PREPULL_XDG_RUNTIME_DIR="${WRAPPER_PREPULL_XDG_RUNTIME_DIR:-$MERGED_SCRATCH_ROOT/xdg-${RUN_SHORT_ID}-prepull}"
  PREPULL_DOCKER_HOST="${WRAPPER_PREPULL_DOCKER_HOST:-$DOCKER_HOST}"
  PREPULL_DOCKER_PIDFILE="${WRAPPER_PREPULL_DOCKER_PIDFILE:-}"
  PREPULL_DOCKER_EXEC_ROOT="${WRAPPER_PREPULL_DOCKER_EXEC_ROOT:-}"
else
  DOCKER_HOST="${WRAPPER_DOCKER_HOST:-unix://$XDG_RUNTIME_DIR/docker.sock}"
  DOCKER_PIDFILE="${WRAPPER_DOCKER_PIDFILE:-$XDG_RUNTIME_DIR/docker.pid}"
  DOCKER_EXEC_ROOT="${WRAPPER_DOCKER_EXEC_ROOT:-$MERGED_SCRATCH_ROOT/${RUN_SHORT_ID}-rootless-exec}"
  PREPULL_XDG_RUNTIME_DIR="${WRAPPER_PREPULL_XDG_RUNTIME_DIR:-$MERGED_SCRATCH_ROOT/xdg-${RUN_SHORT_ID}-prepull}"
  PREPULL_DOCKER_HOST="${WRAPPER_PREPULL_DOCKER_HOST:-unix://$PREPULL_XDG_RUNTIME_DIR/docker.sock}"
  PREPULL_DOCKER_PIDFILE="${WRAPPER_PREPULL_DOCKER_PIDFILE:-$PREPULL_XDG_RUNTIME_DIR/docker.pid}"
  PREPULL_DOCKER_EXEC_ROOT="${WRAPPER_PREPULL_DOCKER_EXEC_ROOT:-$MERGED_SCRATCH_ROOT/${RUN_SHORT_ID}-rootless-exec-prepull}"
fi

HARBOR_SHARED_UV_CACHE_HOST_DIR="${HARBOR_SHARED_UV_CACHE_HOST_DIR:-$MERGED_SCRATCH_ROOT/harbor-uv-cache}"
HARBOR_SHARED_UV_CACHE_ENV_DIR="${HARBOR_SHARED_UV_CACHE_ENV_DIR:-/harbor-shared/uv-cache}"
HARBOR_SHARED_MINI_SWE_TOOL_HOST_HOME="${HARBOR_SHARED_MINI_SWE_TOOL_HOST_HOME:-$MERGED_SCRATCH_ROOT/harbor-mini-swe-home}"
HARBOR_SHARED_MINI_SWE_TOOL_ENV_HOME="${HARBOR_SHARED_MINI_SWE_TOOL_ENV_HOME:-$HARBOR_SHARED_MINI_SWE_TOOL_HOST_HOME}"
HARBOR_SHARED_UV_PYTHON_HOST_DIR="${HARBOR_SHARED_UV_PYTHON_HOST_DIR:-/home/$USER/.local/share/uv/python}"
HARBOR_SHARED_UV_PYTHON_ENV_DIR="${HARBOR_SHARED_UV_PYTHON_ENV_DIR:-$HARBOR_SHARED_UV_PYTHON_HOST_DIR}"
HARBOR_MINI_SWE_AGENT_GIT_REF="${HARBOR_MINI_SWE_AGENT_GIT_REF:-8e8a515fdcecf3a8e45c3909f7f196bfe18ca89a}"
HARBOR_MINI_SWE_AGENT_UV_OFFLINE="${HARBOR_MINI_SWE_AGENT_UV_OFFLINE:-1}"
ROOTLESS_STATE_BACKUP_ROOT="${ROOTLESS_STATE_BACKUP_ROOT:-$MERGED_SCRATCH_ROOT/r2egym-rootless-state-backups}"
AUTO_CLEAN_ROOTLESS_RUNTIME_STATE="${AUTO_CLEAN_ROOTLESS_RUNTIME_STATE:-true}"

SRUN_RETRIES="${SRUN_RETRIES:-5}"
SRUN_RETRY_DELAY_SEC="${SRUN_RETRY_DELAY_SEC:-2}"

TRAINER_NODES=()
TRAINER_JOB_IDS=()
GPU_CLEANUP_NODES=()
GPU_CLEANUP_JOB_IDS=()
GPU_CLEANUP_LABELS=()
ROLLOUT_JOB_ID_EFFECTIVE=""
ROLLOUT_NODE_EFFECTIVE=""
MERGED_IP=""
ROLLOUT_IP=""

require_cmd() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || {
    echo "Missing required command: $cmd" >&2
    exit 1
  }
}

require_env() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "Missing required environment variable: $name" >&2
    exit 1
  fi
}

dedupe_csv() {
  "$PYTHON_BIN" - <<'PY' "$1"
import sys
items = [item.strip() for item in sys.argv[1].split(",") if item.strip()]
seen = []
for item in items:
    if item not in seen:
        seen.append(item)
print(",".join(seen))
PY
}

resolve_data_root() {
  local candidate=""
  for candidate in \
    "/home/$USER/zthunder_yagent/data/harbor" \
    "/home/$USER/zthunder_agent/data/harbor" \
    "$REPO_ROOT/../data/harbor"; do
    if [ -d "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  echo "Could not resolve Harbor data root" >&2
  exit 1
}

DATA_ROOT="${DATA_ROOT:-$(resolve_data_root)}"
TRAIN_DATA="${TRAIN_DATA:-['$DATA_ROOT/r2egym-trivial','$DATA_ROOT/r2egym-easy','$DATA_ROOT/r2egym-medium','$DATA_ROOT/r2egym-hard']}"
EVAL_DATA="${EVAL_DATA:-$TRAIN_DATA}"
MODEL_PATH="${MODEL_PATH:-/data/zy/models/$USER/models/Qwen3-32B}"

trim_csv_array() {
  local -n ref="$1"
  local idx=""
  for idx in "${!ref[@]}"; do
    ref[$idx]="$(printf '%s' "${ref[$idx]}" | xargs)"
  done
}

parse_trainer_specs() {
  local spec=""
  local node=""
  local job_id=""
  [ -n "$TRAINER_NODE_SPECS" ] || {
    echo "TRAINER_NODE_SPECS is required. Format: node:jobid,node:jobid,node:jobid,node:jobid" >&2
    exit 1
  }
  IFS=',' read -r -a raw_specs <<<"$TRAINER_NODE_SPECS"
  trim_csv_array raw_specs
  for spec in "${raw_specs[@]}"; do
    [ -n "$spec" ] || continue
    node="${spec%%:*}"
    job_id="${spec##*:}"
    if [ -z "$node" ] || [ -z "$job_id" ] || [ "$node" = "$job_id" ]; then
      echo "Invalid trainer spec: $spec" >&2
      exit 1
    fi
    TRAINER_NODES+=("$node")
    TRAINER_JOB_IDS+=("$job_id")
  done
  if [ "${#TRAINER_NODES[@]}" -ne 4 ]; then
    echo "Expected exactly 4 trainer specs, got ${#TRAINER_NODES[@]}: ${TRAINER_NODE_SPECS}" >&2
    exit 1
  fi
}

run_on_node() {
  local job_id="$1"
  local node="$2"
  local cpus="$3"
  local gpus="$4"
  shift 4
  local attempt=1
  local rc=0
  local output_file=""
  mkdir -p "$LOCAL_TMP_DIR"
  while [ "$attempt" -le "$SRUN_RETRIES" ]; do
    output_file="$(mktemp -p "$LOCAL_TMP_DIR" srun.XXXXXX)"
    if srun --jobid "$job_id" --overlap --overcommit --immediate=10 -w "$node" --ntasks=1 --nodes=1 --cpus-per-task="$cpus" --gres="gpu:${gpus}" bash -lc "$*" >"$output_file" 2>&1; then
      cat "$output_file"
      rm -f "$output_file"
      return 0
    fi
    rc=$?
    cat "$output_file" >&2
    if ! grep -Eq "Requested nodes are busy|step creation temporarily disabled" "$output_file"; then
      rm -f "$output_file"
      return "$rc"
    fi
    rm -f "$output_file"
    if [ "$attempt" -ge "$SRUN_RETRIES" ]; then
      return "$rc"
    fi
    sleep "$SRUN_RETRY_DELAY_SEC"
    attempt=$((attempt + 1))
  done
  return "$rc"
}

stage_pid_file() {
  printf '%s/%s.client.pid\n' "$STATE_DIR" "$1"
}

stage_pid_is_running() {
  local stage="$1"
  local pid_file=""
  local pid=""
  pid_file="$(stage_pid_file "$stage")"
  [ -f "$pid_file" ] || return 1
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  [ -n "$pid" ] || return 1
  kill -0 "$pid" 2>/dev/null
}

clear_stale_stage_pid() {
  local stage="$1"
  local pid_file=""
  pid_file="$(stage_pid_file "$stage")"
  if [ -f "$pid_file" ] && ! stage_pid_is_running "$stage"; then
    rm -f "$pid_file"
  fi
}

ensure_stage_not_running() {
  local stage="$1"
  clear_stale_stage_pid "$stage"
  if stage_pid_is_running "$stage"; then
    echo "Stage is already running: $stage (client pid $(cat "$(stage_pid_file "$stage")"))" >&2
    return 1
  fi
}

start_detached_client() {
  local stage="$1"
  local log_file="$2"
  shift 2
  local pid_file=""
  local pid=""
  local detached_cmd=""
  mkdir -p "$STATE_DIR"
  pid_file="$(stage_pid_file "$stage")"
  ensure_stage_not_running "$stage"
  printf -v detached_cmd '%q ' "$@"
  setsid bash -lc "exec $detached_cmd" </dev/null >>"$log_file" 2>&1 &
  pid="$!"
  echo "$pid" >"$pid_file"
  sleep 1
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "Failed to start stage client: $stage" >&2
    tail -n 80 "$log_file" >&2 || true
    return 1
  fi
  echo "$stage.client_pid=$pid"
}

stop_stage_client() {
  local stage="$1"
  local pid_file=""
  local pid=""
  local waited=0
  pid_file="$(stage_pid_file "$stage")"
  clear_stale_stage_pid "$stage"
  if [ ! -f "$pid_file" ]; then
    echo "$stage.client_pid=absent"
    return 0
  fi
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    while kill -0 "$pid" 2>/dev/null; do
      if [ "$waited" -ge 15 ]; then
        kill -9 "$pid" 2>/dev/null || true
        break
      fi
      sleep 1
      waited=$((waited + 1))
    done
  fi
  rm -f "$pid_file"
  echo "$stage.cleaned=OK"
}

stop_rootless_docker_runtime() {
  run_on_node "$MERGED_JOB_ID" "$MERGED_NODE" 1 0 "set -euo pipefail
    sock_path='${DOCKER_HOST#unix://}'
    pid=''
    if [ -f '$DOCKER_PIDFILE' ]; then
      pid=\$(cat '$DOCKER_PIDFILE' 2>/dev/null || true)
      if [ -n \"\$pid\" ] && kill -0 \"\$pid\" 2>/dev/null; then
        kill \"\$pid\" >/dev/null 2>&1 || true
        waited=0
        while kill -0 \"\$pid\" 2>/dev/null; do
          if [ \"\$waited\" -ge '$DOCKER_SHUTDOWN_TIMEOUT_SEC' ]; then
            kill -9 \"\$pid\" >/dev/null 2>&1 || true
            break
          fi
          sleep 1
          waited=\$((waited + 1))
        done
      fi
    fi
    pkill -f -- '/scratch/triton_cache/$USER/xdg-${RUN_SHORT_ID}/dockerd-rootless' >/dev/null 2>&1 || true
    pkill -f -- '/scratch/triton_cache/$USER/${RUN_SHORT_ID}-rootless-exec' >/dev/null 2>&1 || true
    rm -f \"\$sock_path\" '$DOCKER_PIDFILE'
    echo 'rootless_docker_runtime.cleaned=OK'"
}

stage_client_status() {
  local stage="$1"
  local pid_file=""
  pid_file="$(stage_pid_file "$stage")"
  clear_stale_stage_pid "$stage"
  if stage_pid_is_running "$stage"; then
    echo "$stage: running (client pid $(cat "$pid_file"))"
    return 0
  fi
  if [ -f "$pid_file" ]; then
    echo "$stage: stale pidfile"
    return 0
  fi
  echo "$stage: stopped"
}

node_ip() {
  local job_id="$1"
  local node="$2"
  run_on_node "$job_id" "$node" 1 0 "
    primary_ip=\$(
      ip -o -4 addr show up scope global \
        | awk '\$2 !~ /^(lo|docker|br-|veth)/ { split(\$4, a, \"/\"); print a[1]; exit }'
    )
    if [ -n \"\$primary_ip\" ]; then
      printf '%s\n' \"\$primary_ip\"
    else
      hostname -I | awk '{print \$1}'
    fi
  "
}

dataset_spec_paths() {
  local raw_spec="$1"
  "$PYTHON_BIN" - <<'PY' "$raw_spec"
import ast
import os
import sys

raw = sys.argv[1]
try:
    parsed = ast.literal_eval(raw)
except Exception:
    parsed = raw

if isinstance(parsed, str):
    items = [parsed]
elif isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
    items = parsed
else:
    raise SystemExit(f"Dataset spec must be a path string or Python list of path strings, got: {raw!r}")

for item in items:
    print(os.path.expanduser(item))
PY
}

resolve_preflight_task_path() {
  local dataset_path="$1"
  "$PYTHON_BIN" - <<'PY' "$dataset_path" "$DATA_ROOT"
import json
import sys
from pathlib import Path

dataset_path = Path(sys.argv[1]).expanduser()
data_root = Path(sys.argv[2]).expanduser()

if dataset_path.is_dir():
    child_dirs = sorted(p for p in dataset_path.iterdir() if p.is_dir())
    if child_dirs:
        print(child_dirs[0])
        raise SystemExit(0)

manifest_path = dataset_path / "MANIFEST.json"
if not manifest_path.is_file():
    raise SystemExit(0)

data = json.loads(manifest_path.read_text())
tasks = data.get("tasks", {})
for bucket in ("trivial", "easy", "medium", "hard"):
    for rel_task in tasks.get(bucket, []):
        candidate = data_root / rel_task
        if candidate.is_dir():
            print(candidate)
            raise SystemExit(0)
PY
}

wait_for_head_docker() {
  local deadline=$((SECONDS + HEAD_DOCKER_READY_TIMEOUT_SEC))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if run_on_node "$MERGED_JOB_ID" "$MERGED_NODE" 1 0 "export DOCKER_HOST='$DOCKER_HOST'; timeout 10 docker info >/dev/null 2>&1"; then
      return 0
    fi
    sleep 2
  done
  echo "Timed out waiting for Docker on $MERGED_NODE via $DOCKER_HOST (mode=$DOCKER_MODE)" >&2
  return 1
}

wait_for_ray() {
  local ray_address="$1"
  local deadline=$((SECONDS + 180))
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
  local deadline=$((SECONDS + ROLLOUT_HEALTH_TIMEOUT_SEC))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if curl -sf "http://$host:$port/health" >/dev/null; then
      return 0
    fi
    clear_stale_stage_pid "rollout"
    if ! stage_pid_is_running "rollout"; then
      echo "Rollout stage client exited before $name became healthy" >&2
      tail -n 80 "$ROLLOUT_LOG" >&2 || true
      return 1
    fi
    sleep 2
  done
  echo "Timed out waiting for $name at http://$host:$port/health" >&2
  tail -n 80 "$ROLLOUT_LOG" >&2 || true
  return 1
}

docker_is_ready() {
  [ -n "$MERGED_NODE" ] || return 1
  [ -n "$MERGED_JOB_ID" ] || return 1
  run_on_node "$MERGED_JOB_ID" "$MERGED_NODE" 1 0 "export DOCKER_HOST='$DOCKER_HOST'; timeout 10 docker info >/dev/null 2>&1" >/dev/null 2>&1
}

ray_is_ready() {
  [ -n "$MERGED_IP" ] || return 1
  "$RAY_BIN" status --address "$MERGED_IP:$RAY_PORT" >/dev/null 2>&1
}

rollout_port_is_ready() {
  local port="$1"
  [ -n "$ROLLOUT_IP" ] || return 1
  curl -sf "http://$ROLLOUT_IP:$port/health" >/dev/null 2>&1
}

ensure_rollout_ready_for_driver() {
  local idx=""
  local suffixes=(a b c d e f g h i j k l m n o p)
  for idx in "${!ROLLOUT_PORTS[@]}"; do
    wait_for_rollout_health "$ROLLOUT_IP" "${ROLLOUT_PORTS[$idx]}" "rollout_${suffixes[$idx]}"
  done
}

run_agent_runtime_preflight() {
  local preflight_script=""
  local target_base=""
  local target_port=""
  local preflight_task_root=""
  local preflight_task_path=""

  [ "$RUN_PREFLIGHT_CHECKS" = true ] || return 0
  [ "$AGENT_RUNTIME_PREFLIGHT" = true ] || return 0

  while IFS= read -r preflight_task_root; do
    [ -n "$preflight_task_root" ] || continue
    if [[ "$preflight_task_root" == *r2egym-trivial* ]]; then
      break
    fi
  done < <(dataset_spec_paths "$TRAIN_DATA")
  if [ -z "$preflight_task_root" ]; then
    preflight_task_root="$(dataset_spec_paths "$TRAIN_DATA" | head -n 1)"
  fi
  preflight_task_path="$(resolve_preflight_task_path "$preflight_task_root")"
  if [ -z "$preflight_task_path" ]; then
    echo "Failed to find a preflight task under $preflight_task_root" >&2
    return 1
  fi

  target_port="${ROLLOUT_PORTS[0]}"
  target_base="http://$ROLLOUT_IP:$target_port/v1"
  read -r -d '' preflight_script <<EOF || true
set -euo pipefail
TMP_ROOT=\$(mktemp -d '$MERGED_SCRATCH_ROOT/agent-preflight-${RUN_SHORT_ID}.XXXXXX')
trap 'rm -rf "\$TMP_ROOT"' EXIT
TOOL_HOME='$HARBOR_SHARED_MINI_SWE_TOOL_HOST_HOME'
TOOL_PY="\$TOOL_HOME/.local/share/uv/tools/mini-swe-agent/bin/python"
TOOL_MINI="\$TOOL_HOME/.local/share/uv/tools/mini-swe-agent/bin/mini"
TARGET_BASE='$target_base'
ROUTER_PORT='$SKYRL_INFERENCE_ROUTER_PORT'
PREFLIGHT_TASK_PATH='$preflight_task_path'
export DOCKER_HOST='$DOCKER_HOST'
export HARBOR_SHARED_UV_CACHE_HOST_DIR='$HARBOR_SHARED_UV_CACHE_HOST_DIR'
export HARBOR_SHARED_UV_CACHE_ENV_DIR='$HARBOR_SHARED_UV_CACHE_ENV_DIR'
export HARBOR_SHARED_MINI_SWE_TOOL_HOST_HOME='$HARBOR_SHARED_MINI_SWE_TOOL_HOST_HOME'
export HARBOR_SHARED_MINI_SWE_TOOL_ENV_HOME='$HARBOR_SHARED_MINI_SWE_TOOL_ENV_HOME'
export HARBOR_SHARED_UV_PYTHON_HOST_DIR='$HARBOR_SHARED_UV_PYTHON_HOST_DIR'
export HARBOR_SHARED_UV_PYTHON_ENV_DIR='$HARBOR_SHARED_UV_PYTHON_ENV_DIR'
export HARBOR_MINI_SWE_AGENT_UV_OFFLINE='$HARBOR_MINI_SWE_AGENT_UV_OFFLINE'
export PREFLIGHT_TASK_PATH
export TMP_ROOT

echo "preflight_step=check_tooling"
[ -x "\$TOOL_PY" ]
[ -x "\$TOOL_MINI" ]
echo "preflight_step=check_router_port_free"
! ss -ltn "( sport = :\$ROUTER_PORT )" | grep -q LISTEN
echo "preflight_step=check_rollout_models"
curl -sf "\$TARGET_BASE/models" >/dev/null

cat >"\$TMP_ROOT/litellm_model_registry.json" <<'JSON'
{
  "hosted_vllm/Qwen3-32B": {
    "max_tokens": 32768,
    "input_cost_per_token": 0.0,
    "output_cost_per_token": 0.0,
    "litellm_provider": "hosted_vllm",
    "mode": "chat"
  },
  "Qwen3-32B": {
    "max_tokens": 32768,
    "input_cost_per_token": 0.0,
    "output_cost_per_token": 0.0,
    "litellm_provider": "hosted_vllm",
    "mode": "chat"
  }
}
JSON

echo "preflight_step=litellm_completion"
HOSTED_VLLM_API_BASE="\$TARGET_BASE" \\
HOSTED_VLLM_API_KEY="fake-api-key" \\
LITELLM_MODEL_REGISTRY_PATH="\$TMP_ROOT/litellm_model_registry.json" \\
timeout 60 "\$TOOL_PY" - <<'PY'
import json
import litellm
import os

with open(os.environ["LITELLM_MODEL_REGISTRY_PATH"]) as f:
    registry = json.load(f)
litellm.register_model(registry)
resp = litellm.completion(
    model="hosted_vllm/Qwen3-32B",
    messages=[{"role": "user", "content": "Reply with exactly ok."}],
    api_base=os.environ["HOSTED_VLLM_API_BASE"],
    api_key=os.environ["HOSTED_VLLM_API_KEY"],
    max_tokens=8,
)
cost = litellm.cost_calculator.completion_cost(resp)
content = (resp.choices[0].message.content or "").strip()
if not content:
    raise SystemExit("litellm preflight returned empty content")
print(f"litellm_preflight_content={content!r}")
print(f"litellm_preflight_cost={cost}")
PY

cat >"\$TMP_ROOT/mini.yaml" <<'YAML'
model:
  model_name: hosted_vllm/Qwen3-32B
  model_kwargs:
    api_base: $target_base
YAML

echo "preflight_step=harbor_trial_reward"
HOSTED_VLLM_API_BASE="\$TARGET_BASE" \\
HOSTED_VLLM_API_KEY="fake-api-key" \\
LITELLM_MODEL_REGISTRY_PATH="\$TMP_ROOT/litellm_model_registry.json" \\
timeout 600 '$PYTHON_BIN' - <<'PY'
import asyncio
import json
import os
from pathlib import Path

import harbor
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig, VerifierConfig


async def main() -> None:
    tmp_root = Path(os.environ["TMP_ROOT"])
    task_path = Path(os.environ["PREFLIGHT_TASK_PATH"])
    config = TrialConfig(
        task=TaskConfig(path=task_path),
        trials_dir=tmp_root / "trial-runs",
        trial_name="agentRuntimePreflight",
        agent=AgentConfig(
            name="mini-swe-agent",
            model_name="hosted_vllm/Qwen3-32B",
            override_timeout_sec=300,
            kwargs={
                "api_base": os.environ["HOSTED_VLLM_API_BASE"],
                "max_turns": 1,
                "enable_summarize": False,
                "record_terminal_session": False,
                "store_all_messages": True,
                "temperature": 0.0,
            },
        ),
        environment=EnvironmentConfig(
            force_build=False,
            delete=True,
            override_cpus=2,
            override_memory_mb=4096,
            override_storage_mb=4096,
        ),
        verifier=VerifierConfig(disable=False),
    )
    trial = harbor.Trial(config)
    result = await trial.run()
    if result.exception_info is not None:
        raise RuntimeError(f"trial preflight exception: {result.exception_info}")
    if result.verifier_result is None or "reward" not in result.verifier_result.rewards:
        raise RuntimeError("trial preflight missing verifier reward")
    verifier_stdout = ""
    if trial._trial_paths.test_stdout_path.exists():
        verifier_stdout = trial._trial_paths.test_stdout_path.read_text(errors="ignore")
    task_metadata_path = task_path / "environment" / "workspace" / "metadata.json"
    if task_metadata_path.is_file() and "metadata.json not found at /workspace/metadata.json" in verifier_stdout:
        raise RuntimeError(
            "trial preflight verifier did not receive /workspace/metadata.json from task environment/workspace"
        )
    if result.agent_result is None or (result.agent_result.n_output_tokens or 0) <= 0:
        raise RuntimeError(
            f"trial preflight missing agent output tokens: {getattr(result.agent_result, 'n_output_tokens', None)}"
        )
    print(
        json.dumps(
            {
                "reward": result.verifier_result.rewards["reward"],
                "n_output_tokens": result.agent_result.n_output_tokens,
                "trial_dir": str(trial.trial_dir),
                "task_path": str(task_path),
            },
            sort_keys=True,
        )
    )


asyncio.run(main())
PY

echo "agent_runtime_preflight=OK"
EOF

  if ! run_on_node "$MERGED_JOB_ID" "$MERGED_NODE" 1 0 "$preflight_script" >"$AGENT_RUNTIME_PREFLIGHT_LOG" 2>&1; then
    tail -n 120 "$AGENT_RUNTIME_PREFLIGHT_LOG" >&2 || true
    return 1
  fi
  cat "$AGENT_RUNTIME_PREFLIGHT_LOG"
}

show_status() {
  local idx=""
  local suffixes=(a b c d e f g h i j k l m n o p)
  echo "run_name=$RUN_NAME"
  echo "log_dir=$LOG_DIR"
  if [ "$DOCKER_MODE" = "rootful" ]; then
    echo "head_docker: external-system-daemon ($DOCKER_HOST)"
  else
    stage_client_status "head_docker"
  fi
  stage_client_status "ray_head"
  for idx in "${!TRAINER_NODES[@]}"; do
    stage_client_status "ray_worker_${idx}"
  done
  stage_client_status "rollout"
  stage_client_status "driver"
  if docker_is_ready; then
    echo "docker_ready=yes"
  else
    echo "docker_ready=no"
  fi
  if ray_is_ready; then
    echo "ray_ready=yes"
  else
    echo "ray_ready=no"
  fi
  if [ -n "$ROLLOUT_IP" ]; then
    for idx in "${!ROLLOUT_PORTS[@]}"; do
      if rollout_port_is_ready "${ROLLOUT_PORTS[$idx]}"; then
        echo "rollout_${suffixes[$idx]}_ready=yes"
      else
        echo "rollout_${suffixes[$idx]}_ready=no"
      fi
    done
  fi
}

cleanup_stage() {
  local stage="$1"
  local idx=""
  case "$stage" in
    head)
      if [ "$DOCKER_MODE" = "rootful" ]; then
        stop_stage_client "head_docker"
        cleanup_rootful_head_harbor_compose_leftovers
        echo "head_docker: external-system-daemon (preserved)"
      else
        stop_stage_client "head_docker"
        stop_rootless_docker_runtime
      fi
      ;;
    ray)
      stop_stage_client "ray_head"
      for idx in "${!TRAINER_NODES[@]}"; do
        stop_stage_client "ray_worker_${idx}"
      done
      ;;
    rollout)
      stop_stage_client "rollout"
      ;;
    driver)
      stop_stage_client "driver"
      cleanup_local_driver_waiters
      ;;
    all)
      stop_stage_client "driver"
      stop_stage_client "rollout"
      cleanup_stage ray
      cleanup_stage head
      ;;
    *)
      echo "Unknown stage for cleanup-stage: $stage" >&2
      return 1
      ;;
  esac
}

cleanup_local_driver_waiters() {
  local pids=""
  local pid=""
  pids="$(pgrep -f -- "wait_harbor_driver_until_terminal.py $TRAIN_DRIVER_LOG" || true)"
  if [ -z "$pids" ]; then
    echo "driver.waiters=absent"
    return 0
  fi
  while IFS= read -r pid; do
    [ -n "$pid" ] || continue
    kill "$pid" 2>/dev/null || true
  done <<<"$pids"
  sleep 1
  while IFS= read -r pid; do
    [ -n "$pid" ] || continue
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  done <<<"$pids"
  echo "driver.waiters.cleaned=OK"
}

cleanup_rootful_head_harbor_compose_leftovers() {
  [ "$DOCKER_MODE" = "rootful" ] || return 0
  run_on_node "$MERGED_JOB_ID" "$MERGED_NODE" 1 0 "set -euo pipefail
    export DOCKER_HOST='$DOCKER_HOST'
    export RUN_ARTIFACT_DIR='$RUN_ARTIFACT_DIR'
    '$PYTHON_BIN' - <<'PY'
import os
import subprocess
from pathlib import Path

run_artifact_dir = Path(os.environ['RUN_ARTIFACT_DIR'])
trials_root = run_artifact_dir / 'trials_run'
trial_names = set()
if trials_root.exists():
    trial_names = {path.name.lower() for path in trials_root.iterdir() if path.is_dir()}


def list_resources(kind: str) -> list[tuple[str, str]]:
    if kind == 'containers':
        cmd = [
            'docker',
            'ps',
            '-a',
            '--format',
            '{{.ID}}\t{{.Label \"com.docker.compose.project\"}}',
        ]
    elif kind == 'networks':
        cmd = [
            'docker',
            'network',
            'ls',
            '--format',
            '{{.ID}}\t{{.Label \"com.docker.compose.project\"}}',
        ]
    else:
        raise ValueError(f'unknown kind: {kind}')
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SystemExit(
            f'failed to list {kind}: returncode={result.returncode} stderr={result.stderr.strip()}'
        )
    rows = []
    for line in result.stdout.splitlines():
        parts = line.rstrip().split('\t', 1)
        if len(parts) != 2:
            continue
        resource_id, project = parts
        if (project or '').lower() in trial_names:
            rows.append((resource_id, project))
    return rows


containers = list_resources('containers')
container_ids = [resource_id for resource_id, _ in containers]
if container_ids:
    result = subprocess.run(
        ['docker', 'rm', '-f', *container_ids],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(
            f'failed to remove containers: returncode={result.returncode} stderr={result.stderr.strip()}'
        )

networks = list_resources('networks')
network_ids = [resource_id for resource_id, _ in networks]
if network_ids:
    result = subprocess.run(
        ['docker', 'network', 'rm', *network_ids],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(
            f'failed to remove networks: returncode={result.returncode} stderr={result.stderr.strip()}'
        )

print(
    'head_harbor_compose_cleanup '
    f'trial_names={len(trial_names)} '
    f'containers_removed={len(container_ids)} '
    f'networks_removed={len(network_ids)}'
)
PY"
}

ensure_rootful_shared_docker_network() {
  [ "$DOCKER_MODE" = "rootful" ] || return 0
  [ -n "$HARBOR_DOCKER_SHARED_NETWORK_NAME" ] || return 0
  run_on_node "$MERGED_JOB_ID" "$MERGED_NODE" 1 0 "set -euo pipefail
    export DOCKER_HOST='$DOCKER_HOST'
    export HARBOR_DOCKER_SHARED_NETWORK_NAME='$HARBOR_DOCKER_SHARED_NETWORK_NAME'
    export HARBOR_DOCKER_SHARED_NETWORK_SUBNET='$HARBOR_DOCKER_SHARED_NETWORK_SUBNET'
    '$PYTHON_BIN' - <<'PY'
import json
import os
import subprocess
import sys

name = os.environ['HARBOR_DOCKER_SHARED_NETWORK_NAME'].strip()
subnet = os.environ.get('HARBOR_DOCKER_SHARED_NETWORK_SUBNET', '').strip()

if not name:
    raise SystemExit(0)

def run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, check=False)

inspect_result = run(['docker', 'network', 'inspect', name])
created = False
if inspect_result.returncode != 0:
    create_cmd = ['docker', 'network', 'create', '--driver', 'bridge', '--attachable']
    if subnet:
        create_cmd.extend(['--subnet', subnet])
    create_cmd.extend(['--label', 'harbor.shared.network=1', name])
    create_result = run(create_cmd)
    if create_result.returncode != 0:
        raise SystemExit(
            f'failed to create shared docker network {name}: '
            f'{create_result.stderr.strip() or create_result.stdout.strip()}'
        )
    created = True
    inspect_result = run(['docker', 'network', 'inspect', name])

if inspect_result.returncode != 0:
    raise SystemExit(
        f'failed to inspect shared docker network {name}: '
        f'{inspect_result.stderr.strip() or inspect_result.stdout.strip()}'
    )

payload = json.loads(inspect_result.stdout)[0]
driver = payload.get('Driver')
actual_subnet = ''
ipam_cfg = payload.get('IPAM', {}).get('Config') or []
if ipam_cfg:
    actual_subnet = ipam_cfg[0].get('Subnet') or ''

if driver != 'bridge':
    raise SystemExit(
        f'shared docker network {name} has unexpected driver={driver!r}'
    )
if subnet and actual_subnet and subnet != actual_subnet:
    raise SystemExit(
        f'shared docker network {name} subnet mismatch: expected={subnet} actual={actual_subnet}'
    )

print(
    'shared_docker_network_ready '
    f'name={name} '
    f'created={created} '
    f'driver={driver} '
    f'subnet={actual_subnet or \"<unset>\"}'
)
PY"
}

run_prepare_stage() {
  ensure_harbor_runtime_patches
  validate_external_assets
  ensure_rootful_shared_docker_network
  backfill_r2egym_docker_images
  if [ "$DOCKER_MODE" = "rootless" ]; then
    validate_rootless_subid_map
  fi
  ensure_kernel_key_quota
  validate_cluster_env
  validate_runtime_scratch_paths
  ensure_clean_gpu_nodes
  if [ "$DOCKER_MODE" = "rootless" ]; then
    cleanup_stale_rootless_runtime_state
  fi
  prepull_r2egym_images
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

ensure_harbor_runtime_patches() {
  if [ "$AUTO_APPLY_HARBOR_RUNTIME_PATCHES" = true ]; then
    "$PYTHON_BIN" "$SCRIPT_DIR/ops/apply_harbor_runtime_patches.py" --backup
  else
    "$PYTHON_BIN" "$SCRIPT_DIR/ops/apply_harbor_runtime_patches.py" --check
  fi
}

validate_external_assets() {
  local path=""

  for path in "$MODEL_PATH/config.json" "$HARBOR_SHARED_UV_PYTHON_HOST_DIR"; do
    if [ ! -e "$path" ]; then
      echo "Required local path missing: $path" >&2
      exit 1
    fi
  done

  while IFS= read -r path; do
    [ -n "$path" ] || continue
    if [ ! -d "$path" ]; then
      echo "Dataset path missing: $path" >&2
      exit 1
    fi
  done < <(dataset_spec_paths "$TRAIN_DATA"; dataset_spec_paths "$EVAL_DATA")

  if [ "$DOCKER_MODE" = "rootful" ]; then
    run_on_node "$MERGED_JOB_ID" "$MERGED_NODE" 1 0 "export DOCKER_HOST='$DOCKER_HOST'; docker compose version >/dev/null && timeout 10 docker info >/dev/null && mkdir -p '$MERGED_SCRATCH_ROOT' '$HARBOR_SHARED_UV_CACHE_HOST_DIR' '$HARBOR_SHARED_MINI_SWE_TOOL_HOST_HOME' '$MERGED_TMP_DIR' && test -w '$MERGED_SCRATCH_ROOT'"
  else
    run_on_node "$MERGED_JOB_ID" "$MERGED_NODE" 1 0 "docker compose version >/dev/null && command -v slirp4netns rootlesskit newuidmap newgidmap dockerd-rootless.sh >/dev/null && mkdir -p '$MERGED_SCRATCH_ROOT' '$DOCKER_DATA_ROOT' '$HARBOR_SHARED_UV_CACHE_HOST_DIR' '$HARBOR_SHARED_MINI_SWE_TOOL_HOST_HOME' '$MERGED_TMP_DIR' && test -w '$MERGED_SCRATCH_ROOT' && test -w '$DOCKER_DATA_ROOT'"
  fi
  run_on_node "$ROLLOUT_JOB_ID_EFFECTIVE" "$ROLLOUT_NODE_EFFECTIVE" 1 0 "[ -f '$MODEL_PATH/config.json' ]"
}

backfill_r2egym_docker_images() {
  local roots_csv=""
  local root=""
  roots_csv="$(while IFS= read -r root; do printf '%s,' "$root"; done < <(dataset_spec_paths "$TRAIN_DATA"; dataset_spec_paths "$EVAL_DATA"))"
  roots_csv="${roots_csv%,}"
  roots_csv="$(dedupe_csv "$roots_csv")"
  [ -n "$roots_csv" ] || return 0
  if [ "$AUTO_BACKFILL_R2EGYM_DOCKER_IMAGES" != true ]; then
    return 0
  fi
  IFS=',' read -r -a roots <<<"$roots_csv"
  "$PYTHON_BIN" "$SCRIPT_DIR/ops/backfill_r2egym_task_docker_image.py" "${roots[@]}"
}

validate_rootless_subid_map() {
  [ "$ROOTLESS_SUBID_CHECK" = true ] || return 0
  run_on_node "$MERGED_JOB_ID" "$MERGED_NODE" 1 0 "'$PYTHON_BIN' - <<'PY'
import os
from pathlib import Path

uid = os.getuid()
gid = os.getgid()

def parse(path: str):
    ranges = []
    for line in Path(path).read_text().splitlines():
        parts = line.split(':')
        if len(parts) != 3 or parts[0] != os.environ['USER']:
            continue
        start = int(parts[1])
        count = int(parts[2])
        ranges.append((start, start + count - 1))
    return ranges

def has_overlap(ranges, value):
    return any(start <= value <= end for start, end in ranges)

uid_ranges = parse('/etc/subuid')
gid_ranges = parse('/etc/subgid')
if has_overlap(uid_ranges, uid):
    raise SystemExit(f'/etc/subuid subordinate range overlaps real uid {uid}: {uid_ranges}')
if has_overlap(gid_ranges, gid):
    raise SystemExit(f'/etc/subgid subordinate range overlaps real gid {gid}: {gid_ranges}')
print('rootless_subid_check=OK')
PY"
}

ensure_kernel_key_quota() {
  [ "$AUTO_RAISE_KERNEL_KEY_QUOTA" = true ] || return 0
  run_on_node "$MERGED_JOB_ID" "$MERGED_NODE" 1 0 "set -euo pipefail
    before_maxkeys=\$(cat /proc/sys/kernel/keys/maxkeys)
    before_maxbytes=\$(cat /proc/sys/kernel/keys/maxbytes)
    before_ses_count=\$(grep -c '_ses\\\\.' /proc/keys || true)
    echo \"kernel_key_quota_before maxkeys=\$before_maxkeys maxbytes=\$before_maxbytes ses_count=\$before_ses_count\"
    if [ \"\$before_maxkeys\" -lt '$KERNEL_KEYS_MAXKEYS' ] || [ \"\$before_maxbytes\" -lt '$KERNEL_KEYS_MAXBYTES' ]; then
      sudo -n sysctl -w kernel.keys.maxkeys='$KERNEL_KEYS_MAXKEYS' kernel.keys.maxbytes='$KERNEL_KEYS_MAXBYTES'
    fi
    after_maxkeys=\$(cat /proc/sys/kernel/keys/maxkeys)
    after_maxbytes=\$(cat /proc/sys/kernel/keys/maxbytes)
    after_ses_count=\$(grep -c '_ses\\\\.' /proc/keys || true)
    echo \"kernel_key_quota_after maxkeys=\$after_maxkeys maxbytes=\$after_maxbytes ses_count=\$after_ses_count\"
    if [ \"\$after_maxkeys\" -lt '$KERNEL_KEYS_MAXKEYS' ] || [ \"\$after_maxbytes\" -lt '$KERNEL_KEYS_MAXBYTES' ]; then
      echo 'Failed to raise kernel key quota to the required floor.' >&2
      exit 1
    fi"
}

run_harbor_docker_concurrency_smoke_prepare() {
  [ "$RUN_HARBOR_DOCKER_CONCURRENCY_SMOKE_PREPARE" = true ] || return 0
  JOB_ID="$MERGED_JOB_ID" \
  HEAD_NODE="$MERGED_NODE" \
  DOCKER_MODE="$DOCKER_MODE" \
  DOCKER_HOST="$DOCKER_HOST" \
  TRIAL_COUNT="$HARBOR_CONCURRENCY_TRIAL_COUNT" \
  MAX_CONCURRENCY="$HARBOR_CONCURRENCY_MAX_IN_FLIGHT" \
  MAX_FAILURES="$HARBOR_CONCURRENCY_MAX_FAILURES" \
  DISABLE_PROJECT_NETWORK="$HARBOR_CONCURRENCY_DISABLE_PROJECT_NETWORK" \
  PYTHON_BIN="$PYTHON_BIN" \
  OUTPUT_ROOT="$LOG_DIR/harbor-docker-concurrency-smoke-prepare" \
  bash "$SCRIPT_DIR/validation/run_harbor_docker_concurrency_smoke.sh"
}

cleanup_stale_rootless_runtime_state() {
  [ "$AUTO_CLEAN_ROOTLESS_RUNTIME_STATE" = true ] || return 0
  run_on_node "$MERGED_JOB_ID" "$MERGED_NODE" 1 0 "set -euo pipefail
    export DOCKER_DATA_ROOT='$DOCKER_DATA_ROOT'
    export ROOTLESS_STATE_BACKUP_ROOT='$ROOTLESS_STATE_BACKUP_ROOT'
    export RUN_TS='$RUN_TS'
    '$PYTHON_BIN' - <<'PY'
from pathlib import Path
import os
import shutil

data_root = Path(os.environ['DOCKER_DATA_ROOT'])
backup_root = Path(os.environ['ROOTLESS_STATE_BACKUP_ROOT']) / os.environ['RUN_TS']
targets = [
    ('containers', 'dir'),
    ('containerd/daemon/io.containerd.runtime.v2.task/moby', 'dir'),
    ('image/overlay2/layerdb/mounts', 'dir'),
    ('network/files/local-kv.db', 'file'),
]

to_backup = []
for rel, kind in targets:
    path = data_root / rel
    if kind == 'dir':
        count = 0
        if path.is_dir():
            count = sum(1 for _ in path.iterdir())
        print(f'{rel}: entries={count}')
        if count > 0:
            to_backup.append((path, rel, kind))
        path.mkdir(parents=True, exist_ok=True)
    else:
        size = path.stat().st_size if path.exists() else 0
        print(f'{rel}: size_bytes={size}')
        if size > 0:
            to_backup.append((path, rel, kind))
        path.parent.mkdir(parents=True, exist_ok=True)

if not to_backup:
    print('rootless_runtime_state_cleanup=SKIPPED')
    raise SystemExit(0)

backup_root.mkdir(parents=True, exist_ok=True)
for path, rel, kind in to_backup:
    dest = backup_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if dest.is_dir():
            shutil.rmtree(dest)
        else:
            dest.unlink()
    shutil.move(str(path), str(dest))
    if kind == 'dir':
        path.mkdir(parents=True, exist_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)

print('rootless_runtime_state_cleanup=APPLIED')
print(f'backup_root={backup_root}')
PY"
}

validate_cluster_env() {
  local all_nodes=("$MERGED_NODE" "${TRAINER_NODES[@]}")
  local all_jobs=("$MERGED_JOB_ID" "${TRAINER_JOB_IDS[@]}")
  local idx=""
  if [ "$ROLLOUT_NODE_EFFECTIVE" != "$MERGED_NODE" ] || [ "$ROLLOUT_JOB_ID_EFFECTIVE" != "$MERGED_JOB_ID" ]; then
    all_nodes+=("$ROLLOUT_NODE_EFFECTIVE")
    all_jobs+=("$ROLLOUT_JOB_ID_EFFECTIVE")
  fi
  for idx in "${!all_nodes[@]}"; do
    run_on_node "${all_jobs[$idx]}" "${all_nodes[$idx]}" 1 0 "cd '$REPO_ROOT' && '$PYTHON_BIN' - <<'PY'
mods = ['ray', 'harbor', 'ThunderAgent', 'torch', 'vllm']
for m in mods:
    __import__(m)
print('imports=OK')
import torch
count = torch.cuda.device_count()
print(f'torch.cuda.device_count={count}')
if count != 8:
    raise SystemExit(f'expected 8 GPUs, got {count}')
PY"
  done
}

validate_runtime_scratch_paths() {
  local trainer_idx=""
  run_on_node "$MERGED_JOB_ID" "$MERGED_NODE" 1 0 \
    "mkdir -p '$MERGED_TMP_DIR' && test -w '$MERGED_TMP_DIR'"
  for trainer_idx in "${!TRAINER_NODES[@]}"; do
    run_on_node "${TRAINER_JOB_IDS[$trainer_idx]}" "${TRAINER_NODES[$trainer_idx]}" 1 0 \
      "mkdir -p '$TRAIN_RUNTIME_SCRATCH_ROOT' '$TRAINER_RAY_TMP_DIR_ROOT/worker_${trainer_idx}' && test -w '$TRAIN_RUNTIME_SCRATCH_ROOT' && test -w '$TRAINER_RAY_TMP_DIR_ROOT/worker_${trainer_idx}'"
  done
  run_on_node "$ROLLOUT_JOB_ID_EFFECTIVE" "$ROLLOUT_NODE_EFFECTIVE" 1 0 \
    "mkdir -p '$ROLLOUT_RUNTIME_SCRATCH_ROOT' && test -w '$ROLLOUT_RUNTIME_SCRATCH_ROOT'"
}

append_gpu_cleanup_target() {
  local job_id="$1"
  local node="$2"
  local label="$3"
  local idx=""
  for idx in "${!GPU_CLEANUP_NODES[@]}"; do
    if [ "${GPU_CLEANUP_NODES[$idx]}" = "$node" ] && [ "${GPU_CLEANUP_JOB_IDS[$idx]}" = "$job_id" ]; then
      GPU_CLEANUP_LABELS[$idx]="${GPU_CLEANUP_LABELS[$idx]},$label"
      return 0
    fi
  done
  GPU_CLEANUP_NODES+=("$node")
  GPU_CLEANUP_JOB_IDS+=("$job_id")
  GPU_CLEANUP_LABELS+=("$label")
}

build_gpu_cleanup_targets() {
  local idx=""
  GPU_CLEANUP_NODES=()
  GPU_CLEANUP_JOB_IDS=()
  GPU_CLEANUP_LABELS=()
  append_gpu_cleanup_target "$MERGED_JOB_ID" "$MERGED_NODE" "merged"
  if [ "$ROLLOUT_GPUS" -gt 0 ]; then
    append_gpu_cleanup_target "$ROLLOUT_JOB_ID_EFFECTIVE" "$ROLLOUT_NODE_EFFECTIVE" "rollout"
  fi
  if [ "$TRAINER_GPUS" -gt 0 ]; then
    for idx in "${!TRAINER_NODES[@]}"; do
      append_gpu_cleanup_target "${TRAINER_JOB_IDS[$idx]}" "${TRAINER_NODES[$idx]}" "trainer_${idx}"
    done
  fi
}

ensure_clean_gpu_nodes() {
  local idx=""
  local node=""
  local job_id=""
  local labels=""
  local log_path=""

  [ "$AUTO_CLEAN_DIRTY_GPUS" = true ] || return 0
  build_gpu_cleanup_targets
  [ "${#GPU_CLEANUP_NODES[@]}" -gt 0 ] || return 0
  mkdir -p "$PREPARE_GPU_LOG_DIR"

  for idx in "${!GPU_CLEANUP_NODES[@]}"; do
    node="${GPU_CLEANUP_NODES[$idx]}"
    job_id="${GPU_CLEANUP_JOB_IDS[$idx]}"
    labels="${GPU_CLEANUP_LABELS[$idx]}"
    log_path="$PREPARE_GPU_LOG_DIR/${node}.log"
    echo "gpu_cleanup_target node=$node job_id=$job_id roles=$labels"
    run_on_node "$job_id" "$node" 1 0 "set -euo pipefail
      export GPU_CLEANUP_LOG_PATH='$log_path'
      export GPU_CLEANUP_LABELS='$labels'
      export GPU_CLEANUP_USE_SUDO='$GPU_CLEANUP_USE_SUDO'
      export GPU_CLEANUP_TERM_TIMEOUT_SEC='$GPU_CLEANUP_TERM_TIMEOUT_SEC'
      export GPU_CLEANUP_KILL_TIMEOUT_SEC='$GPU_CLEANUP_KILL_TIMEOUT_SEC'
      '$PYTHON_BIN' - <<'PY'
from __future__ import annotations

import csv
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

log_path = Path(os.environ['GPU_CLEANUP_LOG_PATH'])
log_path.parent.mkdir(parents=True, exist_ok=True)
labels = os.environ['GPU_CLEANUP_LABELS']
use_sudo = os.environ['GPU_CLEANUP_USE_SUDO'].lower() == 'true'
term_timeout = float(os.environ['GPU_CLEANUP_TERM_TIMEOUT_SEC'])
kill_timeout = float(os.environ['GPU_CLEANUP_KILL_TIMEOUT_SEC'])

with log_path.open('w', encoding='utf-8') as log_fh:
    def emit(line: str = '') -> None:
        print(line, flush=True)
        log_fh.write(line + '\\n')
        log_fh.flush()

    def run_cmd(args: list[str], check: bool = False) -> subprocess.CompletedProcess[str]:
        return subprocess.run(args, check=check, text=True, capture_output=True)

    def query_compute_apps() -> list[dict[str, str]]:
        proc = run_cmd(
            [
                'nvidia-smi',
                '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
                '--format=csv,noheader,nounits',
            ]
        )
        if proc.returncode != 0:
            raise SystemExit(proc.stderr.strip() or proc.stdout.strip() or 'nvidia-smi query failed')
        entries: list[dict[str, str]] = []
        for raw_line in proc.stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            row = next(csv.reader([line], skipinitialspace=True))
            if len(row) < 4:
                continue
            pid = row[1].strip()
            if not pid.isdigit():
                continue
            entries.append(
                {
                    'gpu_uuid': row[0].strip(),
                    'pid': pid,
                    'process_name': row[2].strip(),
                    'used_gpu_memory': row[3].strip(),
                }
            )
        return entries

    def describe(entries: list[dict[str, str]], header: str) -> None:
        emit(header)
        if not entries:
            emit('  clean')
            return
        for entry in entries:
            emit(
                '  gpu_uuid={gpu_uuid} pid={pid} name={process_name} used_gpu_memory_mib={used_gpu_memory}'.format(
                    **entry
                )
            )

    def snapshot_ps(pids: list[str], header: str) -> None:
        if not pids:
            return
        proc = run_cmd(['ps', '-o', 'pid=,user=,ppid=,etime=,cmd=', '-p', ','.join(pids)])
        emit(header)
        stdout = proc.stdout.strip()
        if stdout:
            for line in stdout.splitlines():
                emit(f'  {line.rstrip()}')
        else:
            emit('  ps_output=empty')

    def run_kill(signal_name: str, pids: list[str]) -> None:
        if not pids:
            return
        if use_sudo:
            probe = run_cmd(['sudo', '-n', 'true'])
            if probe.returncode != 0:
                raise SystemExit(probe.stderr.strip() or 'sudo -n true failed')
            proc = run_cmd(['sudo', '-n', 'kill', f'-{signal_name}', *pids])
        else:
            proc = run_cmd(['kill', f'-{signal_name}', *pids])
        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            if stderr:
                emit(f'kill_{signal_name}_stderr={stderr}')

    def wait_until_clean(timeout_sec: float) -> list[dict[str, str]]:
        deadline = time.monotonic() + timeout_sec
        latest = query_compute_apps()
        while latest and time.monotonic() < deadline:
            time.sleep(1)
            latest = query_compute_apps()
        return latest

    node = run_cmd(['hostname', '-s'], check=True).stdout.strip()
    emit(f'node={node}')
    emit(f'roles={labels}')
    emit(f'timestamp_utc={datetime.now(timezone.utc).isoformat()}')
    emit(f'gpu_cleanup_use_sudo={use_sudo}')

    before = query_compute_apps()
    describe(before, '[before]')
    if not before:
        emit('gpu_cleanup_result=clean')
        raise SystemExit(0)

    before_pids = sorted({entry['pid'] for entry in before})
    snapshot_ps(before_pids, '[before ps]')
    emit(f'dirty_pid_count={len(before_pids)}')

    run_kill('TERM', before_pids)
    after_term = wait_until_clean(term_timeout)
    describe(after_term, '[after term]')
    if after_term:
        remaining = sorted({entry['pid'] for entry in after_term})
        snapshot_ps(remaining, '[after term ps]')
        run_kill('KILL', remaining)
    final_entries = wait_until_clean(kill_timeout)
    describe(final_entries, '[after kill]')
    if final_entries:
        final_pids = sorted({entry['pid'] for entry in final_entries})
        snapshot_ps(final_pids, '[after kill ps]')
        raise SystemExit('GPU cleanup failed; remaining_pids={}'.format(' '.join(final_pids)))

    emit('gpu_cleanup_result=clean_after_kill')
PY"
  done
}

prepare_shared_mini_swe_agent_tool_home() {
  mkdir -p "$HARBOR_SHARED_UV_CACHE_HOST_DIR" "$HARBOR_SHARED_MINI_SWE_TOOL_HOST_HOME"
  run_on_node "$MERGED_JOB_ID" "$MERGED_NODE" 1 0 "set -euo pipefail
    export PATH='$REPO_ROOT/.venv/bin':\$HOME/.local/bin:\$PATH
    export HOME='$HARBOR_SHARED_MINI_SWE_TOOL_HOST_HOME'
    export UV_CACHE_DIR='$HARBOR_SHARED_UV_CACHE_HOST_DIR'
    export TMPDIR='$MERGED_TMP_DIR'
    mkdir -p '$HARBOR_SHARED_MINI_SWE_TOOL_HOST_HOME' '$HARBOR_SHARED_UV_CACHE_HOST_DIR' '$MERGED_TMP_DIR'
    uv tool install --offline --cache-dir '$HARBOR_SHARED_UV_CACHE_HOST_DIR' git+https://github.com/li-boxuan/mini-swe-agent.git@'$HARBOR_MINI_SWE_AGENT_GIT_REF' >'$MINI_SWE_TOOL_INSTALL_LOG' 2>&1 || \
    uv tool install --cache-dir '$HARBOR_SHARED_UV_CACHE_HOST_DIR' git+https://github.com/li-boxuan/mini-swe-agent.git@'$HARBOR_MINI_SWE_AGENT_GIT_REF' >>'$MINI_SWE_TOOL_INSTALL_LOG' 2>&1 || {
      cat '$MINI_SWE_TOOL_INSTALL_LOG' >&2
      exit 1
    }"
}

build_r2egym_image_list() {
  local roots=()
  local root=""
  while IFS= read -r root; do
    [ -n "$root" ] || continue
    roots+=("$root")
  done < <(dataset_spec_paths "$TRAIN_DATA")
  "$PYTHON_BIN" - <<'PY' "$IMAGE_LIST_PATH" "${roots[@]}"
from pathlib import Path
import sys
import tomllib

output_path = Path(sys.argv[1])
images = set()
for root_arg in sys.argv[2:]:
    root = Path(root_arg).expanduser()
    for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        task_toml = task_dir / "task.toml"
        parsed = tomllib.loads(task_toml.read_text(encoding="utf-8"))
        docker_image = ((parsed.get("environment") or {}).get("docker_image")) if isinstance(parsed, dict) else None
        if docker_image:
            images.add(docker_image)

output_path.write_text("".join(f"{image}\n" for image in sorted(images)), encoding="utf-8")
print(f"image_count={len(images)}")
PY
  if [ ! -s "$IMAGE_LIST_PATH" ]; then
    echo "Image list is empty after docker_image backfill: $IMAGE_LIST_PATH" >&2
    exit 1
  fi
}

prepull_r2egym_images() {
  [ "$PREPULL_R2EGYM_IMAGES" = true ] || return 0
  require_env DOCKER_USERNAME
  require_env DOCKER_TOKEN
  build_r2egym_image_list
  if [ "$DOCKER_MODE" = "rootful" ]; then
    run_on_node "$MERGED_JOB_ID" "$MERGED_NODE" 8 0 "set -euo pipefail
      export PATH='$REPO_ROOT/.venv/bin':\$HOME/.local/bin:\$PATH
      export TMPDIR='$MERGED_TMP_DIR'
      export DOCKER_HOST='$DOCKER_HOST'
      mkdir -p '$MERGED_TMP_DIR'
      cleanup() {
        docker logout >/dev/null 2>&1 || true
      }
      trap cleanup EXIT
      printf '%s' \"\$DOCKER_TOKEN\" | docker login --username \"\$DOCKER_USERNAME\" --password-stdin >/dev/null
      cat '$IMAGE_LIST_PATH' | xargs -r -n 1 -P '$DOCKER_PULL_CONCURRENCY' docker pull
      echo 'restart_visibility=SKIPPED mode=rootful'
    " >"$PREPULL_LOG" 2>&1
    return 0
  fi
  run_on_node "$MERGED_JOB_ID" "$MERGED_NODE" 8 0 "set -euo pipefail
    export PATH='$REPO_ROOT/.venv/bin':\$HOME/.local/bin:\$PATH
    export TMPDIR='$MERGED_TMP_DIR'
    export XDG_RUNTIME_DIR='$PREPULL_XDG_RUNTIME_DIR'
    export DOCKER_HOST='$PREPULL_DOCKER_HOST'
    export DOCKER_PIDFILE='$PREPULL_DOCKER_PIDFILE'
    export DOCKER_EXEC_ROOT='$PREPULL_DOCKER_EXEC_ROOT'
    export DOCKER_DATA_ROOT='$DOCKER_DATA_ROOT'
    export DOCKER_LOG_PATH='$PREPULL_DOCKER_LOG'
    export SCRATCH_ROOT='$PREPULL_SCRATCH_ROOT'
    export DOCKER_NOFILE_SOFT='$DOCKER_NOFILE_SOFT'
    export DOCKER_INOTIFY_MAX_USER_INSTANCES='$DOCKER_INOTIFY_MAX_USER_INSTANCES'
    export PREPULL_RESTART_VERIFY_VISIBILITY='$PREPULL_RESTART_VERIFY_VISIBILITY'
    export DOCKER_SHUTDOWN_TIMEOUT_SEC='$DOCKER_SHUTDOWN_TIMEOUT_SEC'
    export PREPULL_VERIFY_ROOT_BASE='$PREPULL_VERIFY_ROOT_BASE'
    VERIFY_XDG_RUNTIME_DIR=\"\${VERIFY_XDG_RUNTIME_DIR:-\${PREPULL_VERIFY_ROOT_BASE}/xdg}\"
    VERIFY_DOCKER_HOST=\"\${VERIFY_DOCKER_HOST:-unix://\${VERIFY_XDG_RUNTIME_DIR}/docker.sock}\"
    VERIFY_DOCKER_PIDFILE=\"\${VERIFY_DOCKER_PIDFILE:-\${VERIFY_XDG_RUNTIME_DIR}/docker.pid}\"
    VERIFY_DOCKER_EXEC_ROOT=\"\${VERIFY_DOCKER_EXEC_ROOT:-\${PREPULL_VERIFY_ROOT_BASE}/exec}\"
    VERIFY_SCRATCH_ROOT=\"\${VERIFY_SCRATCH_ROOT:-\${PREPULL_VERIFY_ROOT_BASE}/scratch}\"
    VERIFY_DOCKER_LOG_PATH=\"\${VERIFY_DOCKER_LOG_PATH:-${LOG_DIR}/prepull_rootless_docker.restart_verify.log}\"
    RESTART_VERIFY_MISSING_PATH='${LOG_DIR}/prepull_restart_visibility_missing.txt'
    RESTART_VERIFY_COUNT_PATH='${LOG_DIR}/prepull_restart_visibility_count.txt'
    mkdir -p '$PREPULL_XDG_RUNTIME_DIR' '$PREPULL_DOCKER_EXEC_ROOT' '$DOCKER_DATA_ROOT' '$PREPULL_SCRATCH_ROOT' '$MERGED_TMP_DIR' \"\$VERIFY_XDG_RUNTIME_DIR\" \"\$VERIFY_DOCKER_EXEC_ROOT\" \"\$VERIFY_SCRATCH_ROOT\"
    chmod 700 '$PREPULL_XDG_RUNTIME_DIR' \"\$VERIFY_XDG_RUNTIME_DIR\"
    cd '$REPO_ROOT'
    primary_started_by_script=0
    verify_started_by_script=0

    start_rootless_daemon() {
      local xdg_runtime_dir=\"\$1\"
      local docker_host=\"\$2\"
      local docker_pidfile=\"\$3\"
      local docker_exec_root=\"\$4\"
      local docker_log_path=\"\$5\"
      local scratch_root=\"\$6\"
      mkdir -p \"\$xdg_runtime_dir\" \"\$docker_exec_root\" '$DOCKER_DATA_ROOT' \"\$scratch_root\" '$MERGED_TMP_DIR'
      chmod 700 \"\$xdg_runtime_dir\"
      XDG_RUNTIME_DIR=\"\$xdg_runtime_dir\" \
      DOCKER_HOST=\"\$docker_host\" \
      DOCKER_PIDFILE=\"\$docker_pidfile\" \
      DOCKER_EXEC_ROOT=\"\$docker_exec_root\" \
      DOCKER_DATA_ROOT='$DOCKER_DATA_ROOT' \
      DOCKER_LOG_PATH=\"\$docker_log_path\" \
      SCRATCH_ROOT=\"\$scratch_root\" \
      DOCKER_NOFILE_SOFT='$DOCKER_NOFILE_SOFT' \
      DOCKER_INOTIFY_MAX_USER_INSTANCES='$DOCKER_INOTIFY_MAX_USER_INSTANCES' \
      bash '$SCRIPT_DIR/start_rootless_docker_for_harbor.sh' >/dev/null
    }

    stop_rootless_daemon() {
      local docker_pidfile=\"\$1\"
      local docker_host=\"\$2\"
      local timeout_sec=\"\$3\"
      local waited=0
      local pid=''
      [ -f \"\$docker_pidfile\" ] || return 0
      pid=\"\$(cat \"\$docker_pidfile\" 2>/dev/null || true)\"
      if [ -n \"\$pid\" ] && kill -0 \"\$pid\" 2>/dev/null; then
        kill \"\$pid\" >/dev/null 2>&1 || true
        while kill -0 \"\$pid\" 2>/dev/null; do
          if [ \"\$waited\" -ge \"\$timeout_sec\" ]; then
            kill -9 \"\$pid\" >/dev/null 2>&1 || true
            break
          fi
          sleep 1
          waited=\$((waited + 1))
        done
      fi
      rm -f \"\${docker_host#unix://}\" \"\$docker_pidfile\"
    }

    verify_restart_visibility() {
      local docker_host=\"\$1\"
      local missing_path=\"\$2\"
      : >\"\$missing_path\"
      while IFS= read -r image; do
        [ -n \"\$image\" ] || continue
        if ! DOCKER_HOST=\"\$docker_host\" docker image inspect \"\$image\" >/dev/null 2>&1; then
          printf '%s\n' \"\$image\" >>\"\$missing_path\"
        fi
      done <'$IMAGE_LIST_PATH'
      if [ -s \"\$missing_path\" ]; then
        return 1
      fi
      return 0
    }

    if ! timeout 10 docker info >/dev/null 2>&1; then
      start_rootless_daemon '$PREPULL_XDG_RUNTIME_DIR' '$PREPULL_DOCKER_HOST' '$PREPULL_DOCKER_PIDFILE' '$PREPULL_DOCKER_EXEC_ROOT' '$PREPULL_DOCKER_LOG' '$PREPULL_SCRATCH_ROOT'
      primary_started_by_script=1
    fi

    cleanup() {
      docker logout >/dev/null 2>&1 || true
      if [ \"\$verify_started_by_script\" = 1 ]; then
        stop_rootless_daemon \"\$VERIFY_DOCKER_PIDFILE\" \"\$VERIFY_DOCKER_HOST\" '$DOCKER_SHUTDOWN_TIMEOUT_SEC'
      fi
      if [ \"\$primary_started_by_script\" = 1 ]; then
        stop_rootless_daemon '$PREPULL_DOCKER_PIDFILE' '$PREPULL_DOCKER_HOST' '$DOCKER_SHUTDOWN_TIMEOUT_SEC'
      fi
    }
    trap cleanup EXIT
    printf '%s' \"\$DOCKER_TOKEN\" | docker login --username \"\$DOCKER_USERNAME\" --password-stdin >/dev/null
    cat '$IMAGE_LIST_PATH' | xargs -r -n 1 -P '$DOCKER_PULL_CONCURRENCY' docker pull
    if [ \"\$PREPULL_RESTART_VERIFY_VISIBILITY\" = true ]; then
      if [ \"\$primary_started_by_script\" != 1 ]; then
        echo 'Restart visibility verification requires the prepull step to own the Docker daemon lifecycle for this DOCKER_HOST.' >&2
        echo 'Refusing to stop a pre-existing daemon.' >&2
        exit 1
      fi
      stop_rootless_daemon '$PREPULL_DOCKER_PIDFILE' '$PREPULL_DOCKER_HOST' '$DOCKER_SHUTDOWN_TIMEOUT_SEC'
      primary_started_by_script=0
      start_rootless_daemon \"\$VERIFY_XDG_RUNTIME_DIR\" \"\$VERIFY_DOCKER_HOST\" \"\$VERIFY_DOCKER_PIDFILE\" \"\$VERIFY_DOCKER_EXEC_ROOT\" \"\$VERIFY_DOCKER_LOG_PATH\" \"\$VERIFY_SCRATCH_ROOT\"
      verify_started_by_script=1
      if ! verify_restart_visibility \"\$VERIFY_DOCKER_HOST\" \"\$RESTART_VERIFY_MISSING_PATH\"; then
        echo 'Restart visibility verification failed. Missing tags:' >&2
        cat \"\$RESTART_VERIFY_MISSING_PATH\" >&2
        exit 1
      fi
      DOCKER_HOST=\"\$VERIFY_DOCKER_HOST\" docker images --format '{{.Repository}}:{{.Tag}}' | wc -l >\"\$RESTART_VERIFY_COUNT_PATH\"
      echo \"restart_visibility=OK count=\$(cat \"\$RESTART_VERIFY_COUNT_PATH\")\"
    fi" >"$PREPULL_LOG" 2>&1
}

start_rootless_docker() {
  start_detached_client "head_docker" "$ROOTLESS_DOCKER_LOG" \
    srun --jobid "$MERGED_JOB_ID" --overlap --overcommit --immediate=10 -w "$MERGED_NODE" --ntasks=1 --nodes=1 --cpus-per-task=4 --gres="gpu:0" \
    bash -lc "export PATH='$REPO_ROOT/.venv/bin':\$HOME/.local/bin:\$PATH
      export TMPDIR='$MERGED_TMP_DIR'
      export XDG_RUNTIME_DIR='$XDG_RUNTIME_DIR'
      export DOCKER_HOST='$DOCKER_HOST'
      export DOCKER_PIDFILE='$DOCKER_PIDFILE'
      export DOCKER_EXEC_ROOT='$DOCKER_EXEC_ROOT'
      export DOCKER_DATA_ROOT='$DOCKER_DATA_ROOT'
      export DOCKER_LOG_PATH='$ROOTLESS_DOCKER_LOG'
      export DOCKER_NOFILE_SOFT='$DOCKER_NOFILE_SOFT'
      export DOCKER_INOTIFY_MAX_USER_INSTANCES='$DOCKER_INOTIFY_MAX_USER_INSTANCES'
      export SCRATCH_ROOT='$ROOTLESS_SCRATCH_ROOT'
      mkdir -p '$XDG_RUNTIME_DIR' '$DOCKER_EXEC_ROOT' '$DOCKER_DATA_ROOT' '$ROOTLESS_SCRATCH_ROOT' '$MERGED_TMP_DIR'
      chmod 700 '$XDG_RUNTIME_DIR'
      cd '$REPO_ROOT'
      export ROOTLESS_DOCKER_START_MODE=block
      bash '$SCRIPT_DIR/start_rootless_docker_for_harbor.sh'"
}

start_ray_cluster() {
  local merged_ip="$1"
  local trainer_idx=""
  start_detached_client "ray_head" "$RAY_LOG" \
    srun --jobid "$MERGED_JOB_ID" --overlap --overcommit --immediate=10 -w "$MERGED_NODE" --ntasks=1 --nodes=1 --cpus-per-task=1 --gres="gpu:0" \
    bash -lc "cd '$REPO_ROOT'
      export TMPDIR='$HEAD_RAY_TMP_DIR'
      export RAY_TMPDIR='$HEAD_RAY_TMP_DIR'
      mkdir -p '$HEAD_RAY_TMP_DIR'
      '$RAY_BIN' stop -f >'$LOG_DIR/ray_stop_head.log' 2>&1 || true
      NODE_IP=\$(hostname -I | tr ' ' '\n' | grep '^172\.27\.' | head -n1 || true)
      if [ -z \"\$NODE_IP\" ]; then
        NODE_IP=\$(hostname -I | awk '{print \$1}')
      fi
      HARBOR_HEAD_RESOURCES=\$('${PYTHON_BIN}' -c \"import json; print(json.dumps({'harbor_head': 1}))\")
      '$RAY_BIN' start --head --disable-usage-stats --block --port '$RAY_PORT' --dashboard-host 0.0.0.0 --dashboard-port 8265 --dashboard-agent-listen-port 28280 --dashboard-agent-grpc-port 28380 --runtime-env-agent-port 28480 --min-worker-port 30000 --max-worker-port 30999 --node-ip-address \"\$NODE_IP\" --num-cpus '$HEAD_CPUS' --num-gpus '$HEAD_GPUS' --metrics-export-port 28080 --resources \"\$HARBOR_HEAD_RESOURCES\" --temp-dir '$HEAD_RAY_TMP_DIR'"
  for trainer_idx in "${!TRAINER_NODES[@]}"; do
    start_detached_client "ray_worker_${trainer_idx}" "$RAY_LOG" \
      srun --jobid "${TRAINER_JOB_IDS[$trainer_idx]}" --overlap --overcommit --immediate=10 -w "${TRAINER_NODES[$trainer_idx]}" --ntasks=1 --nodes=1 --cpus-per-task=1 --gres="gpu:0" \
      bash -lc "cd '$REPO_ROOT'
        export TMPDIR='$TRAINER_RAY_TMP_DIR_ROOT/worker_${trainer_idx}'
        export RAY_TMPDIR='$TRAINER_RAY_TMP_DIR_ROOT/worker_${trainer_idx}'
        mkdir -p '$TRAINER_RAY_TMP_DIR_ROOT/worker_${trainer_idx}'
        '$RAY_BIN' stop -f >'$LOG_DIR/ray_stop_worker_${trainer_idx}.log' 2>&1 || true
        NODE_IP=\$(hostname -I | tr ' ' '\n' | grep '^172\.27\.' | head -n1 || true)
        if [ -z \"\$NODE_IP\" ]; then
          NODE_IP=\$(hostname -I | awk '{print \$1}')
        fi
        '$RAY_BIN' start --disable-usage-stats --block --address '$merged_ip:$RAY_PORT' --dashboard-agent-listen-port '$((28200 + trainer_idx))' --dashboard-agent-grpc-port '$((28300 + trainer_idx))' --runtime-env-agent-port '$((28400 + trainer_idx))' --min-worker-port 30000 --max-worker-port 30999 --node-ip-address \"\$NODE_IP\" --num-cpus '$TRAINER_CPUS' --num-gpus '$TRAINER_GPUS' --metrics-export-port '$((28100 + trainer_idx))' --temp-dir '$TRAINER_RAY_TMP_DIR_ROOT/worker_${trainer_idx}'"
  done
}

start_rollout_servers() {
  start_detached_client "rollout" "$ROLLOUT_LOG" \
    srun --jobid "$ROLLOUT_JOB_ID_EFFECTIVE" --overlap --overcommit --immediate=10 -w "$ROLLOUT_NODE_EFFECTIVE" --ntasks=1 --nodes=1 --cpus-per-task="$ROLLOUT_CPUS" --gres="gpu:${ROLLOUT_GPUS}" \
    bash -lc "cd '$REPO_ROOT'
      export PATH='$REPO_ROOT/.venv/bin':\$HOME/.local/bin:\$PATH
      export TMPDIR='$MERGED_TMP_DIR'
      export RUN_NAME='$RUN_NAME'
      export RAY_HEAD_IP='$MERGED_IP'
      export ROLLOUT_SERVER_PORTS_CSV='$ROLLOUT_SERVER_PORTS_CSV'
      export ROLLOUT_GPU_GROUPS_SPEC='$ROLLOUT_GPU_GROUPS_SPEC'
      export TP_SIZE='$ROLLOUT_TP_SIZE'
      export ROLLOUT_NOFILE_SOFT='$ROLLOUT_NOFILE_SOFT'
      export LOG_DIR='$ROLLOUT_LOG_DIR'
      export MONITORING_DIR='$ROLLOUT_MONITOR_DIR'
      export TENSORBOARD_DIR='$ROLLOUT_LOG_DIR/tensorboard'
      export SCRATCH_ROOT='$ROLLOUT_RUNTIME_SCRATCH_ROOT'
      mkdir -p '$MERGED_TMP_DIR'
      bash '$SCRIPT_DIR/start_harbor_rollout_servers.sh'"
}

shell_escape() {
  printf '%q' "$1"
}

build_trainer_resume_overrides() {
  local trainer_resume_overrides=""
  trainer_resume_overrides="trainer.resume_mode='$TRAINER_RESUME_MODE'"
  if [ "$TRAINER_RESUME_MODE" = "from_path" ] && [ -z "$TRAINER_RESUME_PATH" ]; then
    echo "TRAINER_RESUME_PATH is required when TRAINER_RESUME_MODE=from_path" >&2
    exit 1
  fi
  if [ -n "$TRAINER_RESUME_PATH" ]; then
    trainer_resume_overrides="$trainer_resume_overrides trainer.resume_path=$(shell_escape "$TRAINER_RESUME_PATH")"
  fi
  printf '%s\n' "$trainer_resume_overrides"
}

run_training_driver_foreground() {
  local train_data_escaped=""
  local eval_data_escaped=""
  local trainer_resume_overrides=""
  train_data_escaped="$(shell_escape "$TRAIN_DATA")"
  eval_data_escaped="$(shell_escape "$EVAL_DATA")"
  trainer_resume_overrides="$(build_trainer_resume_overrides)"
  set +e
  srun --jobid "$MERGED_JOB_ID" --overlap --overcommit --immediate=10 -w "$MERGED_NODE" --ntasks=1 --nodes=1 --cpus-per-task=8 --gres="gpu:0" \
    bash -lc "cd '$REPO_ROOT'
      export PATH='$PYTHON_BIN_DIR':\$HOME/.local/bin:\$PATH
      export TMPDIR='$MERGED_TMP_DIR'
      export RAY_ADDRESS='$MERGED_IP:$RAY_PORT'
      export RAY_HEAD_IP='$MERGED_IP'
      export ROLLOUT_HOST_IP='$ROLLOUT_IP'
      export SKYRL_INFERENCE_ROUTER_PORT='$SKYRL_INFERENCE_ROUTER_PORT'
      export THUNDER_AGENT_PORT='$SKYRL_INFERENCE_ROUTER_PORT'
      export MSWEA_API_KEY='$MSWEA_API_KEY'
      export TRAIN_DATA=$train_data_escaped
      export EVAL_DATA=$eval_data_escaped
      export ROLLOUT_SERVER_PORTS_CSV='$ROLLOUT_SERVER_PORTS_CSV'
      export ROLLOUT_ENGINES='$ROLLOUT_ENGINES'
      export ROLLOUT_TP_SIZE='$ROLLOUT_TP_SIZE'
      export DOCKER_MODE='$DOCKER_MODE'
      export RUN_NAME_OVERRIDE='$RUN_NAME'
      export LOG_DIR_OVERRIDE='$LOG_DIR'
	      export RUN_ARTIFACT_ROOT='$RUN_ARTIFACT_ROOT'
	      export CKPT_INTERVAL='$CKPT_INTERVAL'
	      export HF_SAVE_INTERVAL='$HF_SAVE_INTERVAL'
	      export CKPT_ROOT_OVERRIDE='$CKPT_ROOT_OVERRIDE'
	      export EXPORT_ROOT_OVERRIDE='$EXPORT_ROOT_OVERRIDE'
	      export SCRATCH_ROOT='$TRAIN_RUNTIME_SCRATCH_ROOT'
	      export XDG_RUNTIME_DIR='$XDG_RUNTIME_DIR'
	      export DOCKER_HOST='$DOCKER_HOST'
      export DOCKER_PIDFILE='$DOCKER_PIDFILE'
      export DOCKER_EXEC_ROOT='$DOCKER_EXEC_ROOT'
      export DOCKER_DATA_ROOT='$DOCKER_DATA_ROOT'
      export HEAD_NOFILE_SOFT='$HEAD_NOFILE_SOFT'
      export HARBOR_SHARED_UV_CACHE_HOST_DIR='$HARBOR_SHARED_UV_CACHE_HOST_DIR'
      export HARBOR_SHARED_UV_CACHE_ENV_DIR='$HARBOR_SHARED_UV_CACHE_ENV_DIR'
      export HARBOR_SHARED_MINI_SWE_TOOL_HOST_HOME='$HARBOR_SHARED_MINI_SWE_TOOL_HOST_HOME'
      export HARBOR_SHARED_MINI_SWE_TOOL_ENV_HOME='$HARBOR_SHARED_MINI_SWE_TOOL_ENV_HOME'
      export HARBOR_SHARED_UV_PYTHON_HOST_DIR='$HARBOR_SHARED_UV_PYTHON_HOST_DIR'
      export HARBOR_SHARED_UV_PYTHON_ENV_DIR='$HARBOR_SHARED_UV_PYTHON_ENV_DIR'
	      export HARBOR_MINI_SWE_AGENT_GIT_REF='$HARBOR_MINI_SWE_AGENT_GIT_REF'
	      export HARBOR_MINI_SWE_AGENT_UV_OFFLINE='$HARBOR_MINI_SWE_AGENT_UV_OFFLINE'
	      export RUN_PREFLIGHT_CHECKS='$RUN_PREFLIGHT_CHECKS'
	      stdbuf -oL -eL bash '$SCRIPT_DIR/run_harbor_fully_async.sh' full max_train_tasks='$MAX_TRAIN_TASKS' max_eval_tasks='$MAX_EVAL_TASKS' trainer.eval_interval='$EVAL_INTERVAL_STEPS' $trainer_resume_overrides harbor_trial_config.agent.name=mini-swe-agent harbor_trial_config.agent.kwargs.max_turns='$HARBOR_AGENT_MAX_TURNS' harbor_trial_config.agent.kwargs.llm_kwargs.timeout='$MINI_SWE_MODEL_TIMEOUT_SEC' harbor_trial_config.agent.override_timeout_sec='$AGENT_TIMEOUT_SEC'" \
    >"$TRAIN_DRIVER_LOG" 2>&1
  TRAIN_RC=$?
  set -e
  return "$TRAIN_RC"
}

start_training_driver_detached() {
  local train_data_escaped=""
  local eval_data_escaped=""
  local trainer_resume_overrides=""
  train_data_escaped="$(shell_escape "$TRAIN_DATA")"
  eval_data_escaped="$(shell_escape "$EVAL_DATA")"
  trainer_resume_overrides="$(build_trainer_resume_overrides)"
  : >"$TRAIN_DRIVER_LOG"
  start_detached_client "driver" "$TRAIN_DRIVER_LOG" \
    srun --jobid "$MERGED_JOB_ID" --overlap --overcommit --immediate=10 -w "$MERGED_NODE" --ntasks=1 --nodes=1 --cpus-per-task=8 --gres="gpu:0" \
    bash -lc "cd '$REPO_ROOT'
      export PATH='$PYTHON_BIN_DIR':\$HOME/.local/bin:\$PATH
      export TMPDIR='$MERGED_TMP_DIR'
      export RAY_ADDRESS='$MERGED_IP:$RAY_PORT'
      export RAY_HEAD_IP='$MERGED_IP'
      export ROLLOUT_HOST_IP='$ROLLOUT_IP'
      export SKYRL_INFERENCE_ROUTER_PORT='$SKYRL_INFERENCE_ROUTER_PORT'
      export THUNDER_AGENT_PORT='$SKYRL_INFERENCE_ROUTER_PORT'
      export MSWEA_API_KEY='$MSWEA_API_KEY'
      export TRAIN_DATA=$train_data_escaped
      export EVAL_DATA=$eval_data_escaped
      export ROLLOUT_SERVER_PORTS_CSV='$ROLLOUT_SERVER_PORTS_CSV'
      export ROLLOUT_ENGINES='$ROLLOUT_ENGINES'
      export ROLLOUT_TP_SIZE='$ROLLOUT_TP_SIZE'
      export DOCKER_MODE='$DOCKER_MODE'
      export RUN_NAME_OVERRIDE='$RUN_NAME'
      export LOG_DIR_OVERRIDE='$LOG_DIR'
	      export RUN_ARTIFACT_ROOT='$RUN_ARTIFACT_ROOT'
	      export CKPT_INTERVAL='$CKPT_INTERVAL'
	      export HF_SAVE_INTERVAL='$HF_SAVE_INTERVAL'
	      export CKPT_ROOT_OVERRIDE='$CKPT_ROOT_OVERRIDE'
	      export EXPORT_ROOT_OVERRIDE='$EXPORT_ROOT_OVERRIDE'
	      export SCRATCH_ROOT='$TRAIN_RUNTIME_SCRATCH_ROOT'
	      export XDG_RUNTIME_DIR='$XDG_RUNTIME_DIR'
	      export DOCKER_HOST='$DOCKER_HOST'
      export DOCKER_PIDFILE='$DOCKER_PIDFILE'
      export DOCKER_EXEC_ROOT='$DOCKER_EXEC_ROOT'
      export DOCKER_DATA_ROOT='$DOCKER_DATA_ROOT'
      export HEAD_NOFILE_SOFT='$HEAD_NOFILE_SOFT'
      export HARBOR_SHARED_UV_CACHE_HOST_DIR='$HARBOR_SHARED_UV_CACHE_HOST_DIR'
      export HARBOR_SHARED_UV_CACHE_ENV_DIR='$HARBOR_SHARED_UV_CACHE_ENV_DIR'
      export HARBOR_SHARED_MINI_SWE_TOOL_HOST_HOME='$HARBOR_SHARED_MINI_SWE_TOOL_HOST_HOME'
      export HARBOR_SHARED_MINI_SWE_TOOL_ENV_HOME='$HARBOR_SHARED_MINI_SWE_TOOL_ENV_HOME'
      export HARBOR_SHARED_UV_PYTHON_HOST_DIR='$HARBOR_SHARED_UV_PYTHON_HOST_DIR'
      export HARBOR_SHARED_UV_PYTHON_ENV_DIR='$HARBOR_SHARED_UV_PYTHON_ENV_DIR'
	      export HARBOR_MINI_SWE_AGENT_GIT_REF='$HARBOR_MINI_SWE_AGENT_GIT_REF'
	      export HARBOR_MINI_SWE_AGENT_UV_OFFLINE='$HARBOR_MINI_SWE_AGENT_UV_OFFLINE'
	      export RUN_PREFLIGHT_CHECKS='$RUN_PREFLIGHT_CHECKS'
	      stdbuf -oL -eL bash '$SCRIPT_DIR/run_harbor_fully_async.sh' full max_train_tasks='$MAX_TRAIN_TASKS' max_eval_tasks='$MAX_EVAL_TASKS' trainer.eval_interval='$EVAL_INTERVAL_STEPS' $trainer_resume_overrides harbor_trial_config.agent.name=mini-swe-agent harbor_trial_config.agent.kwargs.max_turns='$HARBOR_AGENT_MAX_TURNS' harbor_trial_config.agent.kwargs.llm_kwargs.timeout='$MINI_SWE_MODEL_TIMEOUT_SEC' harbor_trial_config.agent.override_timeout_sec='$AGENT_TIMEOUT_SEC'"
}

wait_for_training_driver_terminal() {
  local monitor_dir="$LOG_DIR/monitoring"
  local wait_rc=0
  local -a wait_cmd=(
    "$PYTHON_BIN"
    "$SCRIPT_DIR/ops/wait_harbor_driver_until_terminal.py"
    "$TRAIN_DRIVER_LOG"
    --trial-progress "$monitor_dir/trial_progress.tsv"
    --artifact-dir "$RUN_ARTIFACT_DIR"
    --poll-sec "$DRIVER_TERMINAL_WAIT_POLL_SEC"
    --stale-timeout-sec "$DRIVER_TERMINAL_WAIT_STALE_TIMEOUT_SEC"
    --include-artifacts
  )
  set +e
  "${wait_cmd[@]}"
  wait_rc=$?
  set -e
  clear_stale_stage_pid "driver"
  return "$wait_rc"
}

run_training_driver_canonical() {
  start_training_driver_detached
  wait_for_training_driver_terminal
}

ensure_launch_requirements() {
  require_cmd srun
  require_cmd curl
  require_env MERGED_NODE
  require_env MERGED_JOB_ID
  parse_trainer_specs
  if [ -n "$ROLLOUT_NODE" ] && [ -z "$ROLLOUT_JOB_ID" ]; then
    echo "ROLLOUT_JOB_ID is required when ROLLOUT_NODE is set" >&2
    exit 1
  fi
  if [ -n "$ROLLOUT_JOB_ID" ] && [ -z "$ROLLOUT_NODE" ]; then
    echo "ROLLOUT_NODE is required when ROLLOUT_JOB_ID is set" >&2
    exit 1
  fi
  ROLLOUT_NODE_EFFECTIVE="${ROLLOUT_NODE:-$MERGED_NODE}"
  ROLLOUT_JOB_ID_EFFECTIVE="${ROLLOUT_JOB_ID:-$MERGED_JOB_ID}"
  if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python env not found: $PYTHON_BIN" >&2
    exit 1
  fi
  if [ ! -x "$RAY_BIN" ]; then
    echo "Ray CLI not found: $RAY_BIN" >&2
    exit 1
  fi
  mkdir -p "$LOG_DIR" "$ROLLOUT_LOG_DIR" "$ROLLOUT_MONITOR_DIR" "$RUN_ARTIFACT_DIR" "$LOCAL_TMP_DIR" "$STATE_DIR"
  MERGED_IP="$(node_ip "$MERGED_JOB_ID" "$MERGED_NODE")"
  ROLLOUT_IP="$MERGED_IP"
  if [ "$ROLLOUT_NODE_EFFECTIVE" != "$MERGED_NODE" ] || [ "$ROLLOUT_JOB_ID_EFFECTIVE" != "$MERGED_JOB_ID" ]; then
    ROLLOUT_IP="$(node_ip "$ROLLOUT_JOB_ID_EFFECTIVE" "$ROLLOUT_NODE_EFFECTIVE")"
  fi
  IFS=',' read -r -a ROLLOUT_PORTS <<<"$ROLLOUT_SERVER_PORTS_CSV"
  trim_csv_array ROLLOUT_PORTS
}

print_launch_banner() {
  echo "cross-job topology"
  echo "  action:        $ACTION${ACTION_ARG:+ $ACTION_ARG}"
  echo "  docker_mode:   $DOCKER_MODE"
  echo "  docker_host:   $DOCKER_HOST"
  echo "  merged_node:   $MERGED_NODE (job $MERGED_JOB_ID)"
  echo "  rollout_node:  $ROLLOUT_NODE_EFFECTIVE (job $ROLLOUT_JOB_ID_EFFECTIVE)"
  echo "  trainer_specs: $TRAINER_NODE_SPECS"
  echo "  run_name:      $RUN_NAME"
  echo "  log_dir:       $LOG_DIR"
  echo "  model_path:    $MODEL_PATH"
  echo "  train_data:    $TRAIN_DATA"
  echo "  eval_data:     $EVAL_DATA"
  echo "  max_train:     $MAX_TRAIN_TASKS"
  echo "  max_eval:      $MAX_EVAL_TASKS"
  echo "  full_epochs:   $FULL_EPOCHS"
  echo "  eval_interval: $EVAL_INTERVAL_STEPS"
  echo "  ckpt_root:     ${CKPT_ROOT_OVERRIDE:-<run_artifact_root>}"
  echo "  export_root:   ${EXPORT_ROOT_OVERRIDE:-<run_artifact_root>}"
  echo "  resume_mode:   $TRAINER_RESUME_MODE"
  echo "  resume_path:   ${TRAINER_RESUME_PATH:-<none>}"
}

usage() {
  cat <<EOF
Usage:
  bash $(basename "$0") [all|prepare|head|ray|rollout|driver|driver-preflight|driver-detach|status|cleanup-stage <stage>]

Stages:
  prepare        Run patching, validation, backfill, and optional prepull
  head           Prepare the selected Docker runtime and shared mini-swe-agent tool home
  ray            Start Ray head and 4 trainer workers
  rollout        Start rollout servers and wait for /health
  driver         Start the driver durably and wait for terminal status
  driver-preflight  Run only the driver-side runtime preflight
  driver-detach  Start the driver detached without terminal waiting
  status         Show stage client status and readiness
  cleanup-stage  Stop one stage: head | ray | rollout | driver | all

Notes:
  - Long-lived stages are now launched via detached local srun clients.
  - The canonical driver action now self-detaches locally and waits for terminal driver state.
  - A failure in rollout or driver does not auto-teardown head/ray anymore.
  - Reuse the same RUN_NAME_OVERRIDE when invoking status or cleanup-stage.
  - Canonical benchmark path is cleanup-stage all -> prepare -> head -> ray -> rollout -> status -> driver.
  - Wrapper actions all and driver-detach are blocked for benchmark launches unless ALLOW_UNSAFE_WRAPPER_ACTIONS=true.
EOF
}

ensure_launch_requirements
print_launch_banner

case "$ACTION" in
  all)
    if [ "$ALLOW_UNSAFE_WRAPPER_ACTIONS" != "true" ]; then
      echo "Refusing non-canonical action: all" >&2
      echo "Use the strict benchmark path: cleanup-stage all -> prepare -> head -> ray -> rollout -> status -> driver" >&2
      echo "Set ALLOW_UNSAFE_WRAPPER_ACTIONS=true only if you intentionally want the legacy shortcut." >&2
      exit 1
    fi
    run_prepare_stage
    if [ "$DOCKER_MODE" = "rootless" ]; then
      cleanup_stale_rootless_runtime_state
      start_rootless_docker
    fi
    wait_for_head_docker
    run_harbor_docker_concurrency_smoke_prepare
    prepare_shared_mini_swe_agent_tool_home
    start_ray_cluster "$MERGED_IP"
    wait_for_ray "$MERGED_IP:$RAY_PORT"
    start_rollout_servers
    suffixes=(a b c d e f g h i j k l m n o p)
    for idx in "${!ROLLOUT_PORTS[@]}"; do
      wait_for_rollout_health "$ROLLOUT_IP" "${ROLLOUT_PORTS[$idx]}" "rollout_${suffixes[$idx]}"
    done
    run_training_driver_canonical
    TRAIN_RC=$?
    echo "Run finished with train_rc=$TRAIN_RC"
    echo "  train_driver_log: $TRAIN_DRIVER_LOG"
    if [ "$TRAIN_RC" -eq 0 ] && [ "$CLEANUP_ON_SUCCESS" = true ]; then
      cleanup_stage all
    fi
    exit "$TRAIN_RC"
    ;;
  prepare)
    run_prepare_stage
    ;;
  head)
    ensure_kernel_key_quota
    if [ "$DOCKER_MODE" = "rootless" ]; then
      cleanup_stale_rootless_runtime_state
      start_rootless_docker
    fi
    wait_for_head_docker
    ensure_rootful_shared_docker_network
    run_harbor_docker_concurrency_smoke_prepare
    prepare_shared_mini_swe_agent_tool_home
    ;;
  ray)
    start_ray_cluster "$MERGED_IP"
    wait_for_ray "$MERGED_IP:$RAY_PORT"
    ;;
  rollout)
    start_rollout_servers
    suffixes=(a b c d e f g h i j k l m n o p)
    for idx in "${!ROLLOUT_PORTS[@]}"; do
      wait_for_rollout_health "$ROLLOUT_IP" "${ROLLOUT_PORTS[$idx]}" "rollout_${suffixes[$idx]}"
    done
    ;;
  driver)
    ensure_rollout_ready_for_driver
    run_agent_runtime_preflight
    run_training_driver_canonical
    ;;
  driver-preflight)
    ensure_rollout_ready_for_driver
    run_agent_runtime_preflight
    ;;
  driver-detach)
    if [ "$ALLOW_UNSAFE_WRAPPER_ACTIONS" != "true" ]; then
      echo "Refusing non-canonical action: driver-detach" >&2
      echo "Use the strict benchmark path and launch the driver with: bash \"$0\" driver" >&2
      echo "Set ALLOW_UNSAFE_WRAPPER_ACTIONS=true only if you intentionally want the legacy detached path." >&2
      exit 1
    fi
    ensure_rollout_ready_for_driver
    run_agent_runtime_preflight
    start_training_driver_detached
    ;;
  status)
    show_status
    ;;
  cleanup-stage)
    [ -n "$ACTION_ARG" ] || {
      echo "cleanup-stage requires one of: head | ray | rollout | driver | all" >&2
      exit 1
    }
    cleanup_stage "$ACTION_ARG"
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac
