#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TMP_LOG_ROOT="${TMP_LOG_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/tmp_logs}"

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

ensure_soft_nofile_limit() {
  local requested_soft="$1"
  local current_soft=""
  local current_hard=""

  if [ -z "$requested_soft" ]; then
    return 0
  fi
  if ! [[ "$requested_soft" =~ ^[0-9]+$ ]]; then
    echo "Invalid ROLLOUT_NOFILE_SOFT: $requested_soft" >&2
    exit 1
  fi

  current_soft="$(ulimit -Sn)"
  current_hard="$(ulimit -Hn)"
  if [ "$current_soft" != "unlimited" ] && [ "$current_soft" -lt "$requested_soft" ]; then
    if ! ulimit -Sn "$requested_soft" 2>/dev/null; then
      echo "Failed to raise soft nofile from $current_soft to $requested_soft (hard=$current_hard)." >&2
      echo "Increase the rollout driver's RLIMIT_NOFILE before launching the rollout servers." >&2
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
MODEL_PATH="${MODEL_PATH:-$MODEL_ROOT/Qwen3-32B}"
MODEL_REPO_ID="${MODEL_REPO_ID:-Qwen/Qwen3-32B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-32B}"
ROLLOUT_HOST="${ROLLOUT_HOST:-0.0.0.0}"
PORT_A="${PORT_A:-18000}"
PORT_B="${PORT_B:-18001}"
ROLLOUT_SERVER_PORTS_CSV="${ROLLOUT_SERVER_PORTS_CSV:-}"
ROLLOUT_GPU_GROUPS_SPEC="${ROLLOUT_GPU_GROUPS_SPEC:-}"
TP_SIZE="${TP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
CHAT_TEMPLATE_PATH="${CHAT_TEMPLATE_PATH:-$REPO_ROOT/skyrl/train/utils/templates/qwen3_acc_thinking.jinja2}"
RUN_NAME="${RUN_NAME:-codecontest-qwen3-32b-rollout}"
LOG_DIR="${LOG_DIR:-$TMP_LOG_ROOT/$RUN_NAME/rollout}"
MONITORING_DIR="${MONITORING_DIR:-$LOG_DIR/monitoring}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-$LOG_DIR/tensorboard}"
MONITOR_INTERVAL_SEC="${MONITOR_INTERVAL_SEC:-10}"
AUTO_START_MONITOR="${AUTO_START_MONITOR:-true}"
SCRATCH_ROOT="$(resolve_writable_runtime_root "${SCRATCH_ROOT:-/scratch/$USER/skyrl_runtime}")"
UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH_ROOT/uv-codex}"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$SCRATCH_ROOT/torchinductor}"
TRITON_HOME="${TRITON_HOME:-$SCRATCH_ROOT/triton-home}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SCRATCH_ROOT/triton}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH_ROOT/xdg-cache}"
XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$SCRATCH_ROOT/xdg-config}"
VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$SCRATCH_ROOT/vllm-cache}"
VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-$SCRATCH_ROOT/vllm-config}"
VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-1}"
VLLM_USE_STANDALONE_COMPILE="${VLLM_USE_STANDALONE_COMPILE:-0}"
VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
VLLM_ALLOW_RUNTIME_LORA_UPDATING="${VLLM_ALLOW_RUNTIME_LORA_UPDATING:-true}"
VLLM_ALLOW_INSECURE_SERIALIZATION="${VLLM_ALLOW_INSECURE_SERIALIZATION:-1}"
VLLM_USE_V1="${VLLM_USE_V1:-1}"
VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
ROLLOUT_NOFILE_SOFT="${ROLLOUT_NOFILE_SOFT:-131072}"
HF_ROOT="${HF_ROOT:-/data/zy/models}"
HF_HUB_DIR="${HF_HUB_DIR:-$HF_ROOT/hub}"
HF_XET_DIR="${HF_XET_DIR:-$HF_ROOT/xet}"
PYTHON_BIN="$(resolve_python_bin)"
HF_BIN="$(resolve_hf_bin)"

mkdir -p "$MODEL_ROOT" "$LOG_DIR" "$SCRATCH_ROOT" "$UV_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_HOME" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$VLLM_CACHE_ROOT" "$VLLM_CONFIG_ROOT" "$HF_HUB_DIR" "$HF_XET_DIR"
mkdir -p "$MONITORING_DIR" "$TENSORBOARD_DIR"

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
export VLLM_WORKER_MULTIPROC_METHOD
export VLLM_ALLOW_RUNTIME_LORA_UPDATING
export VLLM_ALLOW_INSECURE_SERIALIZATION
export VLLM_USE_V1
export VLLM_ENABLE_V1_MULTIPROCESSING
export NCCL_CUMEM_ENABLE
export NCCL_P2P_DISABLE
export NCCL_SHM_DISABLE
export OMP_NUM_THREADS
export HF_HOME="$HF_ROOT"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_DIR"
export HF_HUB_CACHE="$HF_HUB_DIR"
export TRANSFORMERS_CACHE="$HF_HUB_DIR"
export HF_XET_CACHE="$HF_XET_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

ensure_soft_nofile_limit "$ROLLOUT_NOFILE_SOFT"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python env not found: $PYTHON_BIN"
  exit 1
fi

SOCKET_IFNAME_TARGET_IP="${SOCKET_IFNAME_TARGET_IP:-${RAY_HEAD_IP:-}}"
DEFAULT_SOCKET_IFNAME="$(detect_socket_ifname "$SOCKET_IFNAME_TARGET_IP")"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$DEFAULT_SOCKET_IFNAME}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$DEFAULT_SOCKET_IFNAME}"

MONITOR_PID=""
SERVER_PIDS=()
SERVER_LOG_FILES=()

server_source_name() {
  local idx="$1"
  local suffixes=(a b c d e f g h i j k l m n o p)
  if [ "$idx" -lt "${#suffixes[@]}" ]; then
    printf 'rollout_%s' "${suffixes[$idx]}"
    return
  fi
  printf 'rollout_%02d' "$idx"
}

trim_csv_array() {
  local -n ref="$1"
  local idx=""
  for idx in "${!ref[@]}"; do
    ref[$idx]="$(printf '%s' "${ref[$idx]}" | xargs)"
  done
}

discover_visible_gpus() {
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    local -a visible=()
    IFS=',' read -r -a visible <<<"$CUDA_VISIBLE_DEVICES"
    trim_csv_array visible
    printf '%s\n' "${visible[@]}"
    return
  fi
  nvidia-smi --query-gpu=index --format=csv,noheader | awk '{$1=$1; print}'
}

build_default_gpu_groups() {
  local server_count="$1"
  local tp_size="$2"
  local -a visible_gpus=()
  local required_gpus=""
  local server_idx=""
  local group_idx=""
  local cursor=0
  local -a group=()
  local joined_group=""

  mapfile -t visible_gpus < <(discover_visible_gpus)
  required_gpus=$((server_count * tp_size))
  if [ "${#visible_gpus[@]}" -lt "$required_gpus" ]; then
    echo "Need at least $required_gpus visible GPUs for $server_count servers with TP=$tp_size, found ${#visible_gpus[@]}" >&2
    return 1
  fi

  for ((server_idx = 0; server_idx < server_count; server_idx++)); do
    group=()
    for ((group_idx = 0; group_idx < tp_size; group_idx++)); do
      group+=("${visible_gpus[$cursor]}")
      cursor=$((cursor + 1))
    done
    joined_group="$(IFS=,; echo "${group[*]}")"
    printf '%s\n' "$joined_group"
  done
}

if [ -n "$ROLLOUT_SERVER_PORTS_CSV" ]; then
  IFS=',' read -r -a SERVER_PORTS <<<"$ROLLOUT_SERVER_PORTS_CSV"
else
  SERVER_PORTS=("$PORT_A" "$PORT_B")
fi
trim_csv_array SERVER_PORTS

if [ -n "$ROLLOUT_GPU_GROUPS_SPEC" ]; then
  IFS=';' read -r -a SERVER_GPU_GROUPS <<<"$ROLLOUT_GPU_GROUPS_SPEC"
  trim_csv_array SERVER_GPU_GROUPS
else
  mapfile -t SERVER_GPU_GROUPS < <(build_default_gpu_groups "${#SERVER_PORTS[@]}" "$TP_SIZE")
fi

if [ "${#SERVER_PORTS[@]}" -eq 0 ]; then
  echo "No rollout server ports configured" >&2
  exit 1
fi
if [ "${#SERVER_GPU_GROUPS[@]}" -ne "${#SERVER_PORTS[@]}" ]; then
  echo "ROLLOUT_GPU_GROUPS_SPEC count (${#SERVER_GPU_GROUPS[@]}) does not match server port count (${#SERVER_PORTS[@]})" >&2
  exit 1
fi

cleanup_monitor() {
  if [ -n "$MONITOR_PID" ]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
  if [ "${#SERVER_PIDS[@]}" -gt 0 ]; then
    kill "${SERVER_PIDS[@]}" 2>/dev/null || true
    wait "${SERVER_PIDS[@]}" 2>/dev/null || true
  fi
}

trap cleanup_monitor EXIT INT TERM

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

cleanup_existing() {
  local port="$1"
  local pids=""
  if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -ti tcp:"$port" || true)"
  elif command -v fuser >/dev/null 2>&1; then
    pids="$(fuser -n tcp "$port" 2>/dev/null || true)"
  fi
  if [ -n "$pids" ]; then
    kill $pids 2>/dev/null || true
    sleep 2
  fi
  if command -v ss >/dev/null 2>&1 && ss -ltn | grep -q ":$port "; then
    echo "Port $port is still listening after cleanup" >&2
    ss -ltnp | grep ":$port " >&2 || true
    return 1
  fi
}

start_server() {
  local gpus="$1"
  local port="$2"
  local log_file="$3"
  local server_idx="$4"

  cleanup_existing "$port"

  SKYRL_EXTERNAL_SERVER_IDX="$server_idx" CUDA_VISIBLE_DEVICES="$gpus" "$PYTHON_BIN" -m skyrl.backends.skyrl_train.inference_engines.vllm.vllm_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --tensor-parallel-size "$TP_SIZE" \
    --host "$ROLLOUT_HOST" \
    --port "$port" \
    --seed 42 \
    --max-model-len "$MAX_MODEL_LEN" \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --dtype bfloat16 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --enable-sleep-mode \
    --max-num_batched_tokens "$MAX_NUM_BATCHED_TOKENS" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --trust-remote-code \
    --chat-template "$CHAT_TEMPLATE_PATH" \
    --distributed-executor-backend mp \
    --worker-extension-cls skyrl.backends.skyrl_train.inference_engines.vllm.vllm_engine.WorkerWrap \
    >"$log_file" 2>&1 &
  LAST_STARTED_PID=$!
}

wait_for_health() {
  local url="$1"
  local name="$2"
  local pid="${3:-}"
  for _ in $(seq 1 900); do
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      echo "$name process exited before becoming healthy" >&2
      return 1
    fi
    if curl -sf "$url/health" >/dev/null; then
      echo "$name healthy at $url"
      return 0
    fi
    sleep 2
  done
  echo "Timed out waiting for $name at $url" >&2
  return 1
}

LAST_STARTED_PID=""

if [ "$AUTO_START_MONITOR" = true ]; then
  MONITOR_ARGS=(
    "$RUN_NAME-rollout"
    "$MONITOR_INTERVAL_SEC"
    "$MONITORING_DIR"
    --tensorboard-dir "$TENSORBOARD_DIR"
    --log-dir "$LOG_DIR"
  )
  for idx in "${!SERVER_PORTS[@]}"; do
    source_name="$(server_source_name "$idx")"
    MONITOR_ARGS+=(--metrics-endpoint "${source_name}.log=http://127.0.0.1:${SERVER_PORTS[$idx]}")
  done
  bash "$SCRIPT_DIR/monitor_stage3_resources.sh" "${MONITOR_ARGS[@]}" &
  MONITOR_PID=$!
  echo "Started rollout monitor pid=$MONITOR_PID output_dir=$MONITORING_DIR"
fi

for idx in "${!SERVER_PORTS[@]}"; do
  cleanup_existing "${SERVER_PORTS[$idx]}"
  source_name="$(server_source_name "$idx")"
  SERVER_LOG_FILES+=("$LOG_DIR/${source_name}.log")
  : >"${SERVER_LOG_FILES[$idx]}"
done

for idx in "${!SERVER_PORTS[@]}"; do
  source_name="$(server_source_name "$idx")"
  start_server "${SERVER_GPU_GROUPS[$idx]}" "${SERVER_PORTS[$idx]}" "${SERVER_LOG_FILES[$idx]}" "$idx"
  SERVER_PIDS+=("$LAST_STARTED_PID")
done

# Launch all rollout engines first so model load and compile overlap across the
# four GPU groups, then gate on health per engine.
for idx in "${!SERVER_PORTS[@]}"; do
  source_name="$(server_source_name "$idx")"
  wait_for_health "http://127.0.0.1:${SERVER_PORTS[$idx]}" "$source_name" "${SERVER_PIDS[$idx]}"
done

echo "External rollout servers ready:"
for idx in "${!SERVER_PORTS[@]}"; do
  source_name="$(server_source_name "$idx")"
  echo "  $source_name: http://$(hostname -f):${SERVER_PORTS[$idx]} (gpus=${SERVER_GPU_GROUPS[$idx]})"
done
echo "Logs:"
for log_file in "${SERVER_LOG_FILES[@]}"; do
  echo "  $log_file"
done

wait "${SERVER_PIDS[@]}"
