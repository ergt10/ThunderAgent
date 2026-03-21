#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VALIDATION_DIR="$SCRIPT_DIR/validation"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"

JOB_ID="${JOB_ID:-}"
VALIDATION_MODE="${VALIDATION_MODE:-full}"
DOCKER_MODE="${DOCKER_MODE:-rootless}"
SRUN_RETRIES="${SRUN_RETRIES:-5}"
SRUN_RETRY_DELAY_SEC="${SRUN_RETRY_DELAY_SEC:-2}"
RAY_PORT="${RAY_PORT:-6381}"
ROLLOUT_PORT_A="${ROLLOUT_PORT_A:-18000}"
ROLLOUT_PORT_B="${ROLLOUT_PORT_B:-18001}"
XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/xdg-test-$USER}"
HEAD_NOFILE_SOFT="${HEAD_NOFILE_SOFT:-131072}"
if [ "$DOCKER_MODE" = rootful ]; then
  DOCKER_HOST="${DOCKER_HOST:-unix:///var/run/docker.sock}"
else
  DOCKER_HOST="${DOCKER_HOST:-unix://$XDG_RUNTIME_DIR/docker.sock}"
fi
STORAGE_ROOT="${STORAGE_ROOT:-/data/zy/models/$USER}"
MODEL_ROOT="${MODEL_ROOT:-$STORAGE_ROOT/models}"
MODEL_PATH="${MODEL_PATH:-$MODEL_ROOT/Qwen3-32B}"
DATA_DIR_DEFAULT="$(cd "$REPO_ROOT/.." && pwd)/data/harbor"
DATA_DIR="${DATA_DIR:-$DATA_DIR_DEFAULT}"
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-$DATA_DIR/CodeContests}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-$DATA_DIR/OpenThoughts-TB-dev}"
TRAIN_DATA="${TRAIN_DATA:-['$TRAIN_DATA_DIR']}"
EVAL_DATA="${EVAL_DATA:-['$EVAL_DATA_DIR']}"
CHAT_TEMPLATE_PATH="${CHAT_TEMPLATE_PATH:-$REPO_ROOT/skyrl/train/utils/templates/qwen3_acc_thinking.jinja2}"
SKIP_ASSET_CHECKS="${SKIP_ASSET_CHECKS:-false}"
SKIP_DOCKER_CHECKS="${SKIP_DOCKER_CHECKS:-false}"
SKIP_ROLLOUT_PORT_CHECKS="${SKIP_ROLLOUT_PORT_CHECKS:-false}"
SKIP_ROLLOUT_HEALTHCHECK="${SKIP_ROLLOUT_HEALTHCHECK:-false}"
SKIP_ROLLOUT_CUDA_RUNTIME_SMOKE="${SKIP_ROLLOUT_CUDA_RUNTIME_SMOKE:-false}"
SKIP_ROOTLESS_HARBOR_CHECK="${SKIP_ROOTLESS_HARBOR_CHECK:-false}"
SKIP_HARBOR_HEAD_CHECK="${SKIP_HARBOR_HEAD_CHECK:-false}"
SKIP_STORAGE_CHECKS="${SKIP_STORAGE_CHECKS:-false}"
SKIP_SOCKET_IFNAME_CHECKS="${SKIP_SOCKET_IFNAME_CHECKS:-false}"
MIN_DATA_FREE_GB="${MIN_DATA_FREE_GB:-100}"
MIN_HOME_FREE_GB="${MIN_HOME_FREE_GB:-20}"
MIN_SCRATCH_FREE_GB="${MIN_SCRATCH_FREE_GB:-200}"

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

check_dataset_spec_on_node() {
  local node="$1"
  local role="$2"
  local raw_spec="$3"
  local path=""
  while IFS= read -r path; do
    [ -n "$path" ] || continue
    run_on_node "$node" "[ -d '$path' ] || { echo '$role dataset missing: $path'; exit 1; }"
  done < <(dataset_spec_paths "$raw_spec")
}

run_on_node() {
  local node="$1"
  shift
  local attempt=1
  local rc=0
  local output_file=""
  local wrapped_cmd="$*"
  if [ -n "${HEAD_NOFILE_SOFT:-}" ]; then
    wrapped_cmd="ulimit -Sn ${HEAD_NOFILE_SOFT} && ${wrapped_cmd}"
  fi
  while [ "$attempt" -le "$SRUN_RETRIES" ]; do
    output_file="$(mktemp)"
    if srun --jobid "$JOB_ID" --overlap --overcommit --immediate=10 -w "$node" --ntasks=1 --nodes=1 --cpus-per-task=1 --gres=gpu:0 bash -lc "$wrapped_cmd" >"$output_file" 2>&1; then
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

node_ip() {
  local node="$1"
  run_on_node "$node" "hostname -I | awk '{print \$1}'"
}

require_cmd squeue
require_cmd scontrol
require_cmd srun
require_env JOB_ID

case "$VALIDATION_MODE" in
  full|train-cluster-only)
    ;;
  *)
    echo "Unsupported VALIDATION_MODE: $VALIDATION_MODE" >&2
    exit 1
    ;;
esac

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python env not found: $PYTHON_BIN" >&2
  exit 1
fi

NODELIST="$(squeue -j "$JOB_ID" -h -o '%N')"
if [ -z "$NODELIST" ]; then
  echo "No active nodes found for job $JOB_ID" >&2
  exit 1
fi

mapfile -t JOB_NODES < <(scontrol show hostnames "$NODELIST")
MIN_NODE_COUNT=6
if [ "$VALIDATION_MODE" = train-cluster-only ]; then
  MIN_NODE_COUNT=5
fi
if [ "${#JOB_NODES[@]}" -lt "$MIN_NODE_COUNT" ]; then
  echo "Expected at least $MIN_NODE_COUNT node(s), got ${#JOB_NODES[@]}: ${JOB_NODES[*]}" >&2
  exit 1
fi

HEAD_NODE="${HEAD_NODE:-${JOB_NODES[0]}}"
ROLLOUT_NODE="${ROLLOUT_NODE:-}"
if [ "$VALIDATION_MODE" = full ] && [ -z "$ROLLOUT_NODE" ]; then
  ROLLOUT_NODE="${JOB_NODES[-1]}"
fi

if [ -n "${TRAINER_NODES_CSV:-}" ]; then
  IFS=',' read -r -a TRAINER_NODES <<<"$TRAINER_NODES_CSV"
else
  TRAINER_NODES=()
  for node in "${JOB_NODES[@]}"; do
    if [ "$node" != "$HEAD_NODE" ] && { [ "$VALIDATION_MODE" != full ] || [ "$node" != "$ROLLOUT_NODE" ]; }; then
      TRAINER_NODES+=("$node")
    fi
  done
fi

if [ "${#TRAINER_NODES[@]}" -ne 4 ]; then
  echo "Expected exactly 4 trainer nodes, got ${#TRAINER_NODES[@]}: ${TRAINER_NODES[*]}" >&2
  exit 1
fi

HEAD_IP="$(node_ip "$HEAD_NODE")"
ROLLOUT_IP=""
ROLLOUT_SERVER_URLS="${ROLLOUT_SERVER_URLS:-}"
if [ "$VALIDATION_MODE" = full ]; then
  if [ -z "$ROLLOUT_NODE" ]; then
    echo "ROLLOUT_NODE is required in full validation mode" >&2
    exit 1
  fi
  ROLLOUT_IP="$(node_ip "$ROLLOUT_NODE")"
  if [ -z "$ROLLOUT_SERVER_URLS" ]; then
    ROLLOUT_SERVER_URLS="['http://${ROLLOUT_IP}:${ROLLOUT_PORT_A}','http://${ROLLOUT_IP}:${ROLLOUT_PORT_B}']"
  fi
else
  SKIP_ROLLOUT_HEALTHCHECK=true
  SKIP_ROLLOUT_PORT_CHECKS=true
fi

echo "Topology"
echo "  validation:    $VALIDATION_MODE"
echo "  job_id:        $JOB_ID"
echo "  head_node:     $HEAD_NODE ($HEAD_IP)"
echo "  trainer_nodes: ${TRAINER_NODES[*]}"
echo "  ray_address:   ${HEAD_IP}:${RAY_PORT}"
echo "  train_data:    $TRAIN_DATA"
echo "  eval_data:     $EVAL_DATA"
if [ "$VALIDATION_MODE" = full ]; then
  echo "  rollout_node:  $ROLLOUT_NODE ($ROLLOUT_IP)"
  echo "  rollout_urls:  $ROLLOUT_SERVER_URLS"
else
  echo "  rollout_node:  skipped"
fi

echo
echo "Dependency checks"
ALL_NODES=("$HEAD_NODE" "${TRAINER_NODES[@]}")
if [ "$VALIDATION_MODE" = full ]; then
  ALL_NODES+=("$ROLLOUT_NODE")
fi
for node in "${ALL_NODES[@]}"; do
  run_on_node "$node" "cd '$REPO_ROOT' && '$PYTHON_BIN' - <<'PY'
mods = ['ray', 'harbor', 'ThunderAgent', 'torch', 'vllm']
for m in mods:
    try:
        __import__(m)
        print(f'{m}: OK')
    except Exception as exc:
        raise SystemExit(f'{m}: FAIL: {exc}')
PY"
done

echo
echo "GPU checks"
for node in "${ALL_NODES[@]}"; do
  run_on_node "$node" "cd '$REPO_ROOT' && '$PYTHON_BIN' - <<'PY'
import torch
count = torch.cuda.device_count()
print(f'torch.cuda.device_count={count}')
if count != 8:
    raise SystemExit(f'expected 8 GPUs, got {count}')
PY"
done

if [ "$VALIDATION_MODE" = full ] && [ "$SKIP_ROLLOUT_CUDA_RUNTIME_SMOKE" != true ]; then
  echo
  echo "Rollout CUDA runtime smoke"
  JOB_ID="$JOB_ID" \
  ROLLOUT_NODE="$ROLLOUT_NODE" \
  PYTHON_BIN="$PYTHON_BIN" \
  SRUN_RETRIES="$SRUN_RETRIES" \
  SRUN_RETRY_DELAY_SEC="$SRUN_RETRY_DELAY_SEC" \
  bash "$VALIDATION_DIR/check_rollout_cuda_runtime_smoke.sh"
fi

if [ "$SKIP_STORAGE_CHECKS" != true ]; then
  echo
  echo "Storage checks"
  for node in "${ALL_NODES[@]}"; do
    run_on_node "$node" "check_free_gb() {
      local mount_point=\"\$1\"
      local min_gb=\"\$2\"
      local available_gb
      available_gb=\$(df -BG \"\$mount_point\" | awk 'NR==2 { gsub(/G/, \"\", \$4); print \$4 }')
      echo \"\$mount_point available_gb=\$available_gb min_gb=\$min_gb\"
      [ -n \"\$available_gb\" ] || { echo \"failed to read df for \$mount_point\"; exit 1; }
      [ \"\$available_gb\" -ge \"\$min_gb\" ] || { echo \"insufficient free space on \$mount_point\"; exit 1; }
    }; check_free_gb /data '$MIN_DATA_FREE_GB'; check_free_gb /home '$MIN_HOME_FREE_GB'; check_free_gb /scratch '$MIN_SCRATCH_FREE_GB'"
  done
fi

if [ "$SKIP_SOCKET_IFNAME_CHECKS" != true ]; then
  echo
  echo "Socket interface checks"
  for node in "${ALL_NODES[@]}"; do
    run_on_node "$node" "iface=\$(bash '$SCRIPT_DIR/detect_socket_ifname.sh'); echo socket_ifname=\$iface; [ -n \"\$iface\" ]"
  done
fi

if [ "$SKIP_ASSET_CHECKS" != true ]; then
  echo
  echo "Asset checks"
  run_on_node "$HEAD_NODE" "[ -x '$REPO_ROOT/.venv/bin/hf' ] || [ -x '$REPO_ROOT/.venv/bin/huggingface-cli' ] || command -v hf >/dev/null 2>&1 || command -v huggingface-cli >/dev/null 2>&1 || { echo 'Hugging Face download CLI missing'; exit 1; }"
  check_dataset_spec_on_node "$HEAD_NODE" "train" "$TRAIN_DATA"
  check_dataset_spec_on_node "$HEAD_NODE" "eval" "$EVAL_DATA"
  run_on_node "$HEAD_NODE" "[ -f '$MODEL_PATH/config.json' ] || { echo 'model missing: $MODEL_PATH'; exit 1; }"
  run_on_node "$HEAD_NODE" "[ -f '$CHAT_TEMPLATE_PATH' ] || { echo 'chat template missing: $CHAT_TEMPLATE_PATH'; exit 1; }"
  if [ "$VALIDATION_MODE" = full ]; then
    run_on_node "$ROLLOUT_NODE" "[ -f '$MODEL_PATH/config.json' ] || { echo 'model missing on rollout node: $MODEL_PATH'; exit 1; }"
  fi
fi

echo
echo "Head service checks"
if [ "$SKIP_DOCKER_CHECKS" = true ]; then
  echo "docker checks skipped"
else
  run_on_node "$HEAD_NODE" "command -v docker >/dev/null 2>&1 || { echo 'docker CLI missing'; exit 1; }"
  run_on_node "$HEAD_NODE" "docker compose version >/dev/null 2>&1 || { echo 'docker compose plugin missing'; exit 1; }"
  if [ "$DOCKER_MODE" = rootless ]; then
    run_on_node "$HEAD_NODE" "command -v slirp4netns >/dev/null 2>&1 || { echo 'slirp4netns missing'; exit 1; }"
    run_on_node "$HEAD_NODE" "command -v rootlesskit >/dev/null 2>&1 || { echo 'rootlesskit missing'; exit 1; }"
    run_on_node "$HEAD_NODE" "command -v newuidmap >/dev/null 2>&1 || { echo 'newuidmap missing'; exit 1; }"
    run_on_node "$HEAD_NODE" "command -v newgidmap >/dev/null 2>&1 || { echo 'newgidmap missing'; exit 1; }"
    run_on_node "$HEAD_NODE" "command -v dockerd-rootless.sh >/dev/null 2>&1 || { echo 'dockerd-rootless.sh missing'; exit 1; }"
  fi
  run_on_node "$HEAD_NODE" "[ -S '${DOCKER_HOST#unix://}' ] || { echo 'docker socket missing at ${DOCKER_HOST#unix://}'; exit 1; }"
  run_on_node "$HEAD_NODE" "DOCKER_HOST='$DOCKER_HOST' docker info >/dev/null"
fi
run_on_node "$HEAD_NODE" "ss -ltn | grep -q ':${RAY_PORT} ' || { echo 'ray head port ${RAY_PORT} is not listening'; exit 1; }"

if [ "$VALIDATION_MODE" = full ]; then
  echo
  echo "Rollout service checks"
  if [ "$SKIP_ROLLOUT_PORT_CHECKS" = true ]; then
    echo "rollout port checks skipped"
  else
    run_on_node "$ROLLOUT_NODE" "ss -ltn | grep -q ':${ROLLOUT_PORT_A} ' || { echo 'rollout port ${ROLLOUT_PORT_A} is not listening'; exit 1; }"
    run_on_node "$ROLLOUT_NODE" "ss -ltn | grep -q ':${ROLLOUT_PORT_B} ' || { echo 'rollout port ${ROLLOUT_PORT_B} is not listening'; exit 1; }"
  fi
fi

echo
echo "Preflight"
PREFLIGHT_ARGS=(
  --ray-address "${HEAD_IP}:${RAY_PORT}"
  --train-num-nodes 4
  --train-gpus-per-node 8
)
if [ -n "$ROLLOUT_SERVER_URLS" ]; then
  PREFLIGHT_ARGS+=(--rollout-server-urls "$ROLLOUT_SERVER_URLS")
fi
if [ "$SKIP_ROLLOUT_HEALTHCHECK" = true ]; then
  PREFLIGHT_ARGS+=(--skip-rollout-health)
fi
if [ "$SKIP_ROOTLESS_HARBOR_CHECK" = true ]; then
  PREFLIGHT_ARGS+=(--skip-rootless-harbor-check)
fi
if [ "$SKIP_HARBOR_HEAD_CHECK" = true ]; then
  PREFLIGHT_ARGS+=(--skip-harbor-head-check)
fi

PREFLIGHT_ARGS_ESCAPED=()
for arg in "${PREFLIGHT_ARGS[@]}"; do
  printf -v _escaped_arg '%q' "$arg"
  PREFLIGHT_ARGS_ESCAPED+=("$_escaped_arg")
done
printf -v PREFLIGHT_RAY_ADDRESS '%q' "${HEAD_IP}:${RAY_PORT}"
printf -v PREFLIGHT_ROLLOUT_URLS '%q' "$ROLLOUT_SERVER_URLS"
printf -v PREFLIGHT_PYTHONPATH '%q' "$REPO_ROOT/ThunderAgent:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
printf -v PREFLIGHT_PYTHON_BIN '%q' "$PYTHON_BIN"
printf -v PREFLIGHT_SCRIPT '%q' "$SCRIPT_DIR/preflight_qwen3_32b_6node_cluster.py"
printf -v PREFLIGHT_REPO_ROOT '%q' "$REPO_ROOT"

run_on_node "$HEAD_NODE" "cd $PREFLIGHT_REPO_ROOT && export RAY_ADDRESS=$PREFLIGHT_RAY_ADDRESS && export ROLLOUT_SERVER_URLS=$PREFLIGHT_ROLLOUT_URLS && export PYTHONPATH=$PREFLIGHT_PYTHONPATH && $PREFLIGHT_PYTHON_BIN $PREFLIGHT_SCRIPT ${PREFLIGHT_ARGS_ESCAPED[*]}"
