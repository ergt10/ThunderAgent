#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Harbor fully-async launcher.
#
# Spec summary:
#   - model family: Qwen3-32B
#   - trainer topology: 4 trainer nodes
#   - rollout: external servers, TP configured by env
#   - runtime details: Docker mode, paths, and cluster IPs come from env
#
# Usage:
#   bash examples/train_integrations/harbor/run_harbor_fully_async.sh full
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
STAGE="${1:-full}"
if [ "$#" -gt 0 ]; then
  shift
fi

DOCKER_MODE="${DOCKER_MODE:-rootless}"

resolve_python_bin() {
  if [ -n "${PYTHON_BIN:-}" ]; then
    printf '%s\n' "$PYTHON_BIN"
    return
  fi
  if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    printf '%s\n' "$REPO_ROOT/.venv/bin/python"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return
  fi
}

resolve_hf_bin() {
  if [ -n "${HF_BIN:-}" ]; then
    printf '%s\n' "$HF_BIN"
    return
  fi
  if [ -x "$REPO_ROOT/.venv/bin/hf" ]; then
    printf '%s\n' "$REPO_ROOT/.venv/bin/hf"
    return
  fi
  if [ -x "$REPO_ROOT/.venv/bin/huggingface-cli" ]; then
    printf '%s\n' "$REPO_ROOT/.venv/bin/huggingface-cli"
    return
  fi
  if command -v hf >/dev/null 2>&1; then
    command -v hf
    return
  fi
  if command -v huggingface-cli >/dev/null 2>&1; then
    command -v huggingface-cli
    return
  fi
}

detect_socket_ifname() {
  local target_ip="${1:-}"
  if [ ! -f "$SCRIPT_DIR/detect_socket_ifname.sh" ]; then
    echo "Socket IFNAME helper is missing: $SCRIPT_DIR/detect_socket_ifname.sh" >&2
    exit 1
  fi
  bash "$SCRIPT_DIR/detect_socket_ifname.sh" "$target_ip"
}

resolve_writable_runtime_root() {
  local preferred="${1:-}"
  local candidate=""
  for candidate in \
    "$preferred" \
    "/tmp/$USER/skyrl_runtime" \
    "$(cd "$REPO_ROOT/.." && pwd)/tmp_runtime/$USER"; do
    [ -n "$candidate" ] || continue
    if mkdir -p "$candidate" >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  echo "Failed to find a writable runtime root" >&2
  exit 1
}

trim_csv_array() {
  local -n ref="$1"
  local idx=""
  for idx in "${!ref[@]}"; do
    ref[$idx]="$(printf '%s' "${ref[$idx]}" | xargs)"
  done
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

ensure_dataset_spec_dirs() {
  local role="$1"
  local raw_spec="$2"
  local path=""
  while IFS= read -r path; do
    [ -n "$path" ] || continue
    if [ ! -d "$path" ]; then
      echo "${role} dataset not found: $path" >&2
      exit 1
    fi
  done < <(dataset_spec_paths "$raw_spec")
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

require_env() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "Missing required environment variable: $name" >&2
    exit 1
  fi
}

ensure_soft_nofile_limit() {
  local requested_soft="$1"
  local current_soft=""
  local current_hard=""

  if [ -z "$requested_soft" ]; then
    return 0
  fi
  if ! [[ "$requested_soft" =~ ^[0-9]+$ ]]; then
    echo "Invalid HEAD_NOFILE_SOFT: $requested_soft" >&2
    exit 1
  fi

  current_soft="$(ulimit -Sn)"
  current_hard="$(ulimit -Hn)"
  if [ "$current_soft" != "unlimited" ] && [ "$current_soft" -lt "$requested_soft" ]; then
    if ! ulimit -Sn "$requested_soft" 2>/dev/null; then
      echo "Failed to raise soft nofile from $current_soft to $requested_soft (hard=$current_hard)." >&2
      echo "Increase the head driver's RLIMIT_NOFILE before launching the full Harbor run." >&2
      exit 1
    fi
  fi

  current_soft="$(ulimit -Sn)"
  if [ "$current_soft" != "unlimited" ] && [ "$current_soft" -lt "$requested_soft" ]; then
    echo "Soft nofile is still below the requested threshold after attempting to raise it: $current_soft < $requested_soft" >&2
    exit 1
  fi
}

STORAGE_ROOT="${STORAGE_ROOT:-/data/zy/models/$USER}"
MODEL_ROOT="${MODEL_ROOT:-$STORAGE_ROOT/models}"
DATA_DIR_DEFAULT="$(cd "$REPO_ROOT/.." && pwd)/data/harbor"
DATA_DIR="${DATA_DIR:-$DATA_DIR_DEFAULT}"
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-$DATA_DIR/CodeContests}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-$DATA_DIR/OpenThoughts-TB-dev}"
TRAIN_DATA="${TRAIN_DATA:-['$TRAIN_DATA_DIR']}"
EVAL_DATA="${EVAL_DATA:-['$EVAL_DATA_DIR']}"

RUN_NAME="${RUN_NAME_OVERRIDE:-codecontest-qwen3-32b-6node-rootless-${STAGE}-flash-attn-memtrace}"
ENTRYPOINT="examples.train_integrations.harbor.entrypoints.main_harbor_thunder_agent_fully_async_head_pinned"
DEFAULT_LOG_ROOT="$(cd "$REPO_ROOT/.." && pwd)/tmp_logs"
LOG_ROOT="${LOG_ROOT:-$DEFAULT_LOG_ROOT}"
LOG_DIR="${LOG_DIR_OVERRIDE:-$LOG_ROOT/$RUN_NAME}"
TENSORBOARD_DIR="$LOG_DIR/tensorboard"
RUN_ARTIFACT_ROOT="${RUN_ARTIFACT_ROOT:-$STORAGE_ROOT/harbor_runs}"
RUN_ARTIFACT_DIR="$RUN_ARTIFACT_ROOT/$RUN_NAME"
TRIALS_DIR="$RUN_ARTIFACT_DIR/trials_run"
CKPT_ROOT_OVERRIDE="${CKPT_ROOT_OVERRIDE:-}"
EXPORT_ROOT_OVERRIDE="${EXPORT_ROOT_OVERRIDE:-}"
if [ -n "$CKPT_ROOT_OVERRIDE" ]; then
  CKPTS_DIR="$CKPT_ROOT_OVERRIDE/$RUN_NAME/ckpts"
else
  CKPTS_DIR="$RUN_ARTIFACT_DIR/ckpts"
fi
if [ -n "$EXPORT_ROOT_OVERRIDE" ]; then
  EXPORTS_DIR="$EXPORT_ROOT_OVERRIDE/$RUN_NAME/exports"
else
  EXPORTS_DIR="$RUN_ARTIFACT_DIR/exports"
fi
MONITORING_DIR="$LOG_DIR/monitoring"
ASYNC_TRACE_PATH="$MONITORING_DIR/async_trace.jsonl"
SCRATCH_ROOT="$(resolve_writable_runtime_root "${SCRATCH_ROOT:-/scratch/$USER/skyrl_runtime}")"
TMP_ROOT="$SCRATCH_ROOT/ray_tmp"
UV_CACHE_DIR="$SCRATCH_ROOT/uv-codex"
TORCHINDUCTOR_CACHE_DIR="$SCRATCH_ROOT/torchinductor"
TRITON_HOME="${TRITON_HOME:-$SCRATCH_ROOT/triton-home}"
TRITON_CACHE_DIR="$SCRATCH_ROOT/triton"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH_ROOT/xdg-cache}"
XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$SCRATCH_ROOT/xdg-config}"
VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$SCRATCH_ROOT/vllm-cache}"
VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-$SCRATCH_ROOT/vllm-config}"
VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-1}"
VLLM_USE_STANDALONE_COMPILE="${VLLM_USE_STANDALONE_COMPILE:-0}"
AIOHTTP_CONNECTOR_LIMIT="${AIOHTTP_CONNECTOR_LIMIT:-2048}"
AIOHTTP_CONNECTOR_LIMIT_PER_HOST="${AIOHTTP_CONNECTOR_LIMIT_PER_HOST:-1024}"
HF_ROOT="${HF_ROOT:-/data/zy/models}"
HF_HUB_DIR="$HF_ROOT/hub"
HF_XET_DIR="$HF_ROOT/xet"

mkdir -p \
  "$MODEL_ROOT" \
  "$TRIALS_DIR" \
  "$CKPTS_DIR" \
  "$EXPORTS_DIR" \
  "$LOG_ROOT" \
  "$LOG_DIR" \
  "$MONITORING_DIR" \
  "$SCRATCH_ROOT" \
  "$TMP_ROOT" \
  "$UV_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$TRITON_HOME" \
  "$TRITON_CACHE_DIR" \
  "$XDG_CACHE_HOME" \
  "$XDG_CONFIG_HOME" \
  "$VLLM_CACHE_ROOT" \
  "$VLLM_CONFIG_ROOT" \
  "$HF_HUB_DIR" \
  "$HF_XET_DIR"

export TMPDIR="$TMP_ROOT"
export RAY_TMPDIR="$TMP_ROOT"
export UV_CACHE_DIR
export TORCHINDUCTOR_CACHE_DIR
export TRITON_HOME
export TRITON_CACHE_DIR
export XDG_CACHE_HOME
export XDG_CONFIG_HOME
export VLLM_CACHE_ROOT
export VLLM_CONFIG_ROOT
export VLLM_DISABLE_COMPILE_CACHE
export VLLM_USE_STANDALONE_COMPILE
export AIOHTTP_CONNECTOR_LIMIT
export AIOHTTP_CONNECTOR_LIMIT_PER_HOST
export TENSORBOARD_DIR
export HF_HOME="$HF_ROOT"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_DIR"
export HF_HUB_CACHE="$HF_HUB_DIR"
export TRANSFORMERS_CACHE="$HF_HUB_DIR"
export HF_XET_CACHE="$HF_XET_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SKYRL_ASYNC_TRACE_PATH="$ASYNC_TRACE_PATH"
export _SKYRL_USE_NEW_INFERENCE=1
export SKYRL_PYTHONPATH_EXPORT=1
RAY_PORT="${RAY_PORT:-6381}"
SKYRL_INFERENCE_ROUTER_PORT="${SKYRL_INFERENCE_ROUTER_PORT:-${THUNDER_AGENT_PORT:-8080}}"
THUNDER_AGENT_PORT="${THUNDER_AGENT_PORT:-$SKYRL_INFERENCE_ROUTER_PORT}"
export SKYRL_INFERENCE_ROUTER_PORT
export THUNDER_AGENT_PORT
if [ -n "${RAY_HEAD_IP:-}" ] && [ -z "${RAY_ADDRESS:-}" ]; then
  export RAY_ADDRESS="${RAY_HEAD_IP}:${RAY_PORT}"
else
  export RAY_ADDRESS="${RAY_ADDRESS:-}"
fi
if [ -n "${THUNDERAGENT_URL:-}" ]; then
  export THUNDERAGENT_URL
elif [ -n "${RAY_HEAD_IP:-}" ]; then
  export THUNDERAGENT_URL="http://${RAY_HEAD_IP}:${THUNDER_AGENT_PORT}"
fi
export SKYRL_TRIALS_ROOT="$TRIALS_DIR"
export HARBOR_DOCKER_KEEP_IMAGES="${HARBOR_DOCKER_KEEP_IMAGES:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/xdg-test-$USER}"
if [ "$DOCKER_MODE" = rootful ]; then
  export DOCKER_HOST="${DOCKER_HOST:-unix:///var/run/docker.sock}"
else
  export DOCKER_HOST="${DOCKER_HOST:-unix://$XDG_RUNTIME_DIR/docker.sock}"
fi
export DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT:-/scratch/$USER/docker-rootless}"
HEAD_NOFILE_SOFT="${HEAD_NOFILE_SOFT:-131072}"

MODEL_PATH="${MODEL_PATH:-$MODEL_ROOT/Qwen3-32B}"
MODEL_REPO_ID="${MODEL_REPO_ID:-Qwen/Qwen3-32B}"
MODEL_NAME="${MODEL_NAME:-Qwen3-32B}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
TRAIN_MAX_SEQ_LEN="${TRAIN_MAX_SEQ_LEN:-6144}"
CHAT_TEMPLATE_PATH="${CHAT_TEMPLATE_PATH:-$REPO_ROOT/skyrl/train/utils/templates/qwen3_acc_thinking.jinja2}"
PYTHON_BIN="$(resolve_python_bin)"
HF_BIN="$(resolve_hf_bin)"
PYTHONPATH_PREPEND="${PYTHONPATH_PREPEND:-$REPO_ROOT/ThunderAgent}"

ensure_soft_nofile_limit "$HEAD_NOFILE_SOFT"
echo "Configured head nofile: soft=$(ulimit -Sn) hard=$(ulimit -Hn)"
echo "Configured inference router port: $SKYRL_INFERENCE_ROUTER_PORT"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python env not found: $PYTHON_BIN"
  exit 1
fi

extract_first_rollout_host() {
  local raw_urls="$1"
  if [ -z "$raw_urls" ]; then
    return 0
  fi
  "$PYTHON_BIN" - <<'PY' "$raw_urls"
import ast
import sys
from urllib.parse import urlparse

raw = sys.argv[1]
urls = ast.literal_eval(raw)
if not urls:
    raise SystemExit(0)
print(urlparse(urls[0]).hostname or "")
PY
}

SOCKET_IFNAME_TARGET_IP="${SOCKET_IFNAME_TARGET_IP:-}"
if [ -z "$SOCKET_IFNAME_TARGET_IP" ]; then
  if [ -n "${ROLLOUT_HOST_IP:-}" ]; then
    SOCKET_IFNAME_TARGET_IP="$ROLLOUT_HOST_IP"
  elif [ -n "${RAY_HEAD_IP:-}" ]; then
    SOCKET_IFNAME_TARGET_IP="$RAY_HEAD_IP"
  elif [ -n "$ROLLOUT_SERVER_URLS" ]; then
    SOCKET_IFNAME_TARGET_IP="$(extract_first_rollout_host "$ROLLOUT_SERVER_URLS")"
  elif [ -n "$RAY_ADDRESS" ]; then
    SOCKET_IFNAME_TARGET_IP="${RAY_ADDRESS%%:*}"
  fi
fi

DEFAULT_SOCKET_IFNAME="$(detect_socket_ifname "$SOCKET_IFNAME_TARGET_IP")"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$DEFAULT_SOCKET_IFNAME}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$DEFAULT_SOCKET_IFNAME}"

ensure_model_path() {
  if [ -f "$MODEL_PATH/config.json" ]; then
    return 0
  fi

  if [ -z "$HF_BIN" ] || [ ! -x "$HF_BIN" ]; then
    echo "Hugging Face download CLI not found. Expected one of: $REPO_ROOT/.venv/bin/hf, $REPO_ROOT/.venv/bin/huggingface-cli, hf, huggingface-cli" >&2
    exit 1
  fi

  rm -rf "$MODEL_PATH"
  mkdir -p "$MODEL_PATH"
  HF_HUB_OFFLINE= TRANSFORMERS_OFFLINE= "$HF_BIN" download "$MODEL_REPO_ID" \
    --type model \
    --cache-dir "$HF_HUB_DIR" \
    --local-dir "$MODEL_PATH" \
    --max-workers 16
}

ensure_model_path

ensure_dataset_spec_dirs "Train" "$TRAIN_DATA"
ensure_dataset_spec_dirs "Eval" "$EVAL_DATA"

if [ ! -S "${DOCKER_HOST#unix://}" ]; then
  echo "Docker socket not found at $DOCKER_HOST"
  exit 1
fi

PYTHONPATH_PARTS=("$PYTHONPATH_PREPEND" "$REPO_ROOT")
if [ -n "${PYTHONPATH:-}" ]; then
  PYTHONPATH_PARTS+=("$PYTHONPATH")
fi
export PYTHONPATH="$(IFS=:; echo "${PYTHONPATH_PARTS[*]}")"

TRAIN_NUM_NODES="${TRAIN_NUM_NODES:-4}"
TRAIN_GPUS_PER_NODE="${TRAIN_GPUS_PER_NODE:-8}"
TRAINING_WORLD_SIZE=$((TRAIN_NUM_NODES * TRAIN_GPUS_PER_NODE))
SCALED_SMOKE_BATCH="${SCALED_SMOKE_BATCH:-$((TRAINING_WORLD_SIZE / 2))}"
SCALED_FULL_BATCH="${SCALED_FULL_BATCH:-$TRAINING_WORLD_SIZE}"
FULL_TRAIN_BATCH_SIZE="${FULL_TRAIN_BATCH_SIZE:-64}"
FULL_POLICY_MINI_BATCH_SIZE="${FULL_POLICY_MINI_BATCH_SIZE:-64}"
FULL_MICRO_FORWARD_BATCH_SIZE_PER_GPU="${FULL_MICRO_FORWARD_BATCH_SIZE_PER_GPU:-4}"
FULL_MICRO_TRAIN_BATCH_SIZE_PER_GPU="${FULL_MICRO_TRAIN_BATCH_SIZE_PER_GPU:-4}"
FULL_EPOCHS="${FULL_EPOCHS:-1}"
FULL_N_SAMPLES="${FULL_N_SAMPLES:-4}"
FULL_NUM_PARALLEL_GENERATION_WORKERS="${FULL_NUM_PARALLEL_GENERATION_WORKERS:-64}"
FULL_MAX_CONCURRENCY="${FULL_MAX_CONCURRENCY:-256}"
FULL_TRAJ_PER_SEC="${FULL_TRAJ_PER_SEC:-2}"
FULL_MAX_STALENESS_STEPS="${FULL_MAX_STALENESS_STEPS:-2}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-4}"
ROLLOUT_PORT_A="${ROLLOUT_PORT_A:-18000}"
ROLLOUT_PORT_B="${ROLLOUT_PORT_B:-18001}"
ROLLOUT_SERVER_PORTS_CSV="${ROLLOUT_SERVER_PORTS_CSV:-${ROLLOUT_PORT_A},${ROLLOUT_PORT_B}}"
IFS=',' read -r -a ROLLOUT_PORTS <<<"$ROLLOUT_SERVER_PORTS_CSV"
trim_csv_array ROLLOUT_PORTS
ROLLOUT_ENGINES="${ROLLOUT_ENGINES:-${#ROLLOUT_PORTS[@]}}"
ROLLOUT_METRICS_ENDPOINT_SPECS="${ROLLOUT_METRICS_ENDPOINT_SPECS:-}"
if [ -n "${ROLLOUT_HOST_IP:-}" ] && [ -z "${ROLLOUT_SERVER_URLS:-}" ]; then
  ROLLOUT_SERVER_URLS="$(build_rollout_server_urls_literal "$ROLLOUT_HOST_IP" "${ROLLOUT_PORTS[@]}")"
else
  ROLLOUT_SERVER_URLS="${ROLLOUT_SERVER_URLS:-}"
fi
CKPT_INTERVAL="${CKPT_INTERVAL:-5}"
HF_SAVE_INTERVAL="${HF_SAVE_INTERVAL:-5}"

LOSS_REDUCTION="${LOSS_REDUCTION:-seq_mean_token_sum_norm}"
GRPO_NORM_BY_STD="${GRPO_NORM_BY_STD:-false}"
USE_KL_LOSS="${USE_KL_LOSS:-true}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
TIS_TYPE="${TIS_TYPE:-token}"
TIS_IMP_RATIO_CAP="${TIS_IMP_RATIO_CAP:-2.0}"
APPLY_OVERLONG_FILTERING="${APPLY_OVERLONG_FILTERING:-true}"
MONITOR_INTERVAL_SEC="${MONITOR_INTERVAL_SEC:-10}"
MONITOR_PID=""
FLASH_ATTN="${FLASH_ATTN:-true}"
POLICY_RECORD_MEMORY="${POLICY_RECORD_MEMORY:-true}"
USE_SAMPLE_PACKING="${USE_SAMPLE_PACKING:-false}"
AGENT_TEMPERATURE="${AGENT_TEMPERATURE:-0.3}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"
INCLUDE_REASONING="${INCLUDE_REASONING:-false}"
THUNDER_AGENT_MODE="${THUNDER_AGENT_MODE:-tr}"
THUNDER_AGENT_PROFILE_ENABLED="${THUNDER_AGENT_PROFILE_ENABLED:-true}"
THUNDER_AGENT_METRICS_ENABLED="${THUNDER_AGENT_METRICS_ENABLED:-true}"
RUN_PREFLIGHT_CHECKS="${RUN_PREFLIGHT_CHECKS:-true}"
PREFLIGHT_ROLLOUT_TIMEOUT_SEC="${PREFLIGHT_ROLLOUT_TIMEOUT_SEC:-5}"
SKIP_ROLLOUT_HEALTHCHECK="${SKIP_ROLLOUT_HEALTHCHECK:-false}"
SKIP_HARBOR_ROOTLESS_PATCH_CHECK="${SKIP_HARBOR_ROOTLESS_PATCH_CHECK:-false}"
SKIP_HARBOR_HEAD_CHECK="${SKIP_HARBOR_HEAD_CHECK:-false}"

cleanup_monitor() {
  if [ -n "$MONITOR_PID" ]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
}

trap cleanup_monitor EXIT INT TERM

case "$STAGE" in
  smoke)
    MAX_TRAIN_TASKS="$SCALED_SMOKE_BATCH"
    MAX_EVAL_TASKS=4
    N_SAMPLES=2
    EVAL_N_SAMPLES=1
    TRAIN_BATCH_SIZE="$SCALED_SMOKE_BATCH"
    MINI_BATCH_SIZE="$SCALED_SMOKE_BATCH"
    MICRO_FORWARD_BATCH_SIZE_PER_GPU=1
    MICRO_TRAIN_BATCH_SIZE_PER_GPU=1
    EPOCHS=1
    EVAL_BEFORE_TRAIN=false
    EVAL_INTERVAL=0
    TRAJ_PER_SEC=1
    MAX_CONCURRENCY=2
    LOGGER=console
    COLLECT_MEMORY_METRICS=false
    AUTO_START_MONITOR=false
    MAX_TURNS=8
    TIMEOUT_SEC=900
    MAX_STALENESS_STEPS=1
    NUM_PARALLEL_GENERATION_WORKERS="$SCALED_SMOKE_BATCH"
    ROLLOUT_ENFORCE_EAGER=true
    ;;
  pilot)
    MAX_TRAIN_TASKS=64
    MAX_EVAL_TASKS=10
    N_SAMPLES=2
    EVAL_N_SAMPLES=1
    TRAIN_BATCH_SIZE="$SCALED_FULL_BATCH"
    MINI_BATCH_SIZE="$SCALED_FULL_BATCH"
    MICRO_FORWARD_BATCH_SIZE_PER_GPU=1
    MICRO_TRAIN_BATCH_SIZE_PER_GPU=1
    EPOCHS=1
    EVAL_BEFORE_TRAIN=false
    EVAL_INTERVAL=20
    TRAJ_PER_SEC=1
    MAX_CONCURRENCY=4
    LOGGER=console
    COLLECT_MEMORY_METRICS=false
    AUTO_START_MONITOR=false
    MAX_TURNS=10
    TIMEOUT_SEC=900
    MAX_STALENESS_STEPS=1
    NUM_PARALLEL_GENERATION_WORKERS="$SCALED_FULL_BATCH"
    ROLLOUT_ENFORCE_EAGER=true
    ;;
  full)
    MAX_TRAIN_TASKS=null
    MAX_EVAL_TASKS=20
    N_SAMPLES="$FULL_N_SAMPLES"
    EVAL_N_SAMPLES=1
    TRAIN_BATCH_SIZE="$FULL_TRAIN_BATCH_SIZE"
    MINI_BATCH_SIZE="$FULL_POLICY_MINI_BATCH_SIZE"
    MICRO_FORWARD_BATCH_SIZE_PER_GPU="$FULL_MICRO_FORWARD_BATCH_SIZE_PER_GPU"
    MICRO_TRAIN_BATCH_SIZE_PER_GPU="$FULL_MICRO_TRAIN_BATCH_SIZE_PER_GPU"
    EPOCHS="$FULL_EPOCHS"
    EVAL_BEFORE_TRAIN=false
    EVAL_INTERVAL=50
    TRAJ_PER_SEC="$FULL_TRAJ_PER_SEC"
    MAX_CONCURRENCY="$FULL_MAX_CONCURRENCY"
    LOGGER="['tensorboard','console']"
    COLLECT_MEMORY_METRICS=true
    AUTO_START_MONITOR=true
    MAX_TURNS=10
    TIMEOUT_SEC=900
    MAX_STALENESS_STEPS="$FULL_MAX_STALENESS_STEPS"
    NUM_PARALLEL_GENERATION_WORKERS="$FULL_NUM_PARALLEL_GENERATION_WORKERS"
    ROLLOUT_ENFORCE_EAGER=false
    ;;
  *)
    echo "Usage: $0 {smoke|pilot|full}"
    exit 1
    ;;
esac

if [ "$STAGE" = "full" ] && [ -n "${WANDB_API_KEY:-}" ]; then
  LOGGER="['wandb','tensorboard','console']"
fi

if [ "$AUTO_START_MONITOR" = true ]; then
  MONITOR_ARGS=(
    "$RUN_NAME"
    "$MONITOR_INTERVAL_SEC"
    "$MONITORING_DIR"
  )
  if [ -n "$ROLLOUT_METRICS_ENDPOINT_SPECS" ]; then
    IFS=';' read -r -a ROLLOUT_MONITOR_SPECS <<<"$ROLLOUT_METRICS_ENDPOINT_SPECS"
    trim_csv_array ROLLOUT_MONITOR_SPECS
    for spec in "${ROLLOUT_MONITOR_SPECS[@]}"; do
      [ -n "$spec" ] || continue
      MONITOR_ARGS+=(--metrics-endpoint "$spec")
    done
  elif [ -n "${ROLLOUT_HOST_IP:-}" ]; then
    for idx in "${!ROLLOUT_PORTS[@]}"; do
      source_name="$(server_source_name "$idx")"
      MONITOR_ARGS+=(--metrics-endpoint "${source_name}.log=http://${ROLLOUT_HOST_IP}:${ROLLOUT_PORTS[$idx]}")
    done
  fi
  bash "$SCRIPT_DIR/monitor_stage3_resources.sh" "${MONITOR_ARGS[@]}" &
  MONITOR_PID=$!
  echo "Started fully-async monitor pid=$MONITOR_PID output_dir=$MONITORING_DIR tensorboard_dir=$TENSORBOARD_DIR"
fi

require_env RAY_ADDRESS
require_env ROLLOUT_SERVER_URLS

if [ "$RUN_PREFLIGHT_CHECKS" = true ]; then
  PREFLIGHT_ARGS=(
    "$SCRIPT_DIR/preflight_qwen3_32b_6node_cluster.py"
    --ray-address "$RAY_ADDRESS"
    --train-num-nodes "$TRAIN_NUM_NODES"
    --train-gpus-per-node "$TRAIN_GPUS_PER_NODE"
    --rollout-server-urls "$ROLLOUT_SERVER_URLS"
    --rollout-timeout-sec "$PREFLIGHT_ROLLOUT_TIMEOUT_SEC"
    --min-head-nofile "$HEAD_NOFILE_SOFT"
  )
  if [ "$SKIP_ROLLOUT_HEALTHCHECK" = true ]; then
    PREFLIGHT_ARGS+=(--skip-rollout-health)
  fi
  if [ "$SKIP_HARBOR_ROOTLESS_PATCH_CHECK" = true ]; then
    PREFLIGHT_ARGS+=(--skip-rootless-harbor-check)
  fi
  if [ "$SKIP_HARBOR_HEAD_CHECK" = true ]; then
    PREFLIGHT_ARGS+=(--skip-harbor-head-check)
  fi

  "$PYTHON_BIN" "${PREFLIGHT_ARGS[@]}"
fi

"$PYTHON_BIN" -m "$ENTRYPOINT" \
  data.train_data="$TRAIN_DATA" \
  data.val_data="$EVAL_DATA" \
  max_train_tasks="$MAX_TRAIN_TASKS" \
  max_eval_tasks="$MAX_EVAL_TASKS" \
  trainer.policy.model.path="$MODEL_PATH" \
  generator.inference_engine.served_model_name="$MODEL_NAME" \
  harbor_trial_config.trials_dir="$TRIALS_DIR" \
  trainer.export_path="$EXPORTS_DIR" \
  trainer.ckpt_path="$CKPTS_DIR" \
  trainer.log_path="$LOG_DIR" \
  trainer.strategy=fsdp2 \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.off_policy_correction.tis_ratio_type="$TIS_TYPE" \
  trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high="$TIS_IMP_RATIO_CAP" \
  trainer.algorithm.loss_reduction="$LOSS_REDUCTION" \
  trainer.algorithm.grpo_norm_by_std="$GRPO_NORM_BY_STD" \
  trainer.algorithm.use_kl_loss="$USE_KL_LOSS" \
  trainer.algorithm.kl_loss_coef="$KL_LOSS_COEF" \
  trainer.algorithm.max_seq_len="$TRAIN_MAX_SEQ_LEN" \
  trainer.fully_async.max_staleness_steps="$MAX_STALENESS_STEPS" \
  trainer.fully_async.num_parallel_generation_workers="$NUM_PARALLEL_GENERATION_WORKERS" \
  trainer.placement.colocate_all=false \
  trainer.placement.colocate_policy_ref=true \
  trainer.placement.policy_num_nodes="$TRAIN_NUM_NODES" \
  trainer.placement.policy_num_gpus_per_node="$TRAIN_GPUS_PER_NODE" \
  trainer.placement.ref_num_nodes="$TRAIN_NUM_NODES" \
  trainer.placement.ref_num_gpus_per_node="$TRAIN_GPUS_PER_NODE" \
  trainer.critic.model.path=null \
  trainer.epochs="$EPOCHS" \
  trainer.eval_batch_size=128 \
  trainer.eval_before_train="$EVAL_BEFORE_TRAIN" \
  trainer.eval_interval="$EVAL_INTERVAL" \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size="$TRAIN_BATCH_SIZE" \
  trainer.policy_mini_batch_size="$MINI_BATCH_SIZE" \
  trainer.micro_forward_batch_size_per_gpu="$MICRO_FORWARD_BATCH_SIZE_PER_GPU" \
  trainer.micro_train_batch_size_per_gpu="$MICRO_TRAIN_BATCH_SIZE_PER_GPU" \
  trainer.flash_attn="$FLASH_ATTN" \
  trainer.policy.record_memory="$POLICY_RECORD_MEMORY" \
  trainer.use_sample_packing="$USE_SAMPLE_PACKING" \
  trainer.ckpt_interval="$CKPT_INTERVAL" \
  trainer.hf_save_interval="$HF_SAVE_INTERVAL" \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  generator.n_samples_per_prompt="$N_SAMPLES" \
  generator.eval_n_samples_per_prompt="$EVAL_N_SAMPLES" \
  generator.apply_overlong_filtering="$APPLY_OVERLONG_FILTERING" \
  generator.sampling_params.temperature="$AGENT_TEMPERATURE" \
  generator.sampling_params.logprobs=1 \
  generator.eval_sampling_params.temperature=0.0 \
  generator.eval_sampling_params.logprobs=1 \
  generator.inference_engine.num_engines="$ROLLOUT_ENGINES" \
  generator.inference_engine.tensor_parallel_size="$ROLLOUT_TP_SIZE" \
  generator.inference_engine.run_engines_locally=false \
  generator.inference_engine.external_server_urls="$ROLLOUT_SERVER_URLS" \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.enforce_eager="$ROLLOUT_ENFORCE_EAGER" \
  generator.inference_engine.engine_init_kwargs.chat_template="$CHAT_TEMPLATE_PATH" \
  generator.inference_engine.engine_init_kwargs.max_model_len="$MAX_MODEL_LEN" \
  generator.batched=false \
  generator.rate_limit.enabled=true \
  generator.rate_limit.trajectories_per_second="$TRAJ_PER_SEC" \
  generator.rate_limit.max_concurrency="$MAX_CONCURRENCY" \
  generator.inference_engine.thunder_agent_mode="$THUNDER_AGENT_MODE" \
  generator.inference_engine.thunder_agent_profile_enabled="$THUNDER_AGENT_PROFILE_ENABLED" \
  generator.inference_engine.thunder_agent_metrics_enabled="$THUNDER_AGENT_METRICS_ENABLED" \
  harbor_trial_config.environment.type=docker \
  harbor_trial_config.environment.override_cpus=2 \
  harbor_trial_config.environment.override_memory_mb=4096 \
  harbor_trial_config.environment.override_storage_mb=4096 \
  harbor_trial_config.environment.kwargs.auto_stop_interval_mins=null \
  harbor_trial_config.agent.override_timeout_sec="$TIMEOUT_SEC" \
  harbor_trial_config.agent.kwargs.max_turns="$MAX_TURNS" \
  harbor_trial_config.agent.kwargs.enable_summarize=false \
  harbor_trial_config.agent.kwargs.record_terminal_session=false \
  harbor_trial_config.agent.kwargs.store_all_messages=true \
  harbor_trial_config.agent.kwargs.temperature="$AGENT_TEMPERATURE" \
  harbor_trial_config.agent.kwargs.llm_call_kwargs.extra_body.chat_template_kwargs.enable_thinking="$ENABLE_THINKING" \
  harbor_trial_config.agent.kwargs.llm_call_kwargs.extra_body.include_reasoning="$INCLUDE_REASONING" \
  harbor_trial_config.agent.kwargs.model_info.max_input_tokens="$MAX_MODEL_LEN" \
  harbor_trial_config.agent.kwargs.model_info.max_output_tokens="$MAX_MODEL_LEN" \
  trainer.logger="$LOGGER" \
  trainer.collect_memory_metrics="$COLLECT_MEMORY_METRICS" \
  trainer.collect_memory_metrics_interval=1 \
  trainer.project_name=harbor \
  trainer.run_name="$RUN_NAME" \
  trainer.resume_mode=latest \
  "$@"
