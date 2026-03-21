#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OPS_DIR="$SCRIPT_DIR/ops"
VALIDATION_DIR="$SCRIPT_DIR/validation"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"

JOB_ID="${JOB_ID:-}"
HEAD_NODE="${HEAD_NODE:-}"
ROLLOUT_NODE="${ROLLOUT_NODE:-}"
TRAINER_NODES_CSV="${TRAINER_NODES_CSV:-}"
STAGE="${STAGE:-full}"
DOCKER_MODE="${DOCKER_MODE:-rootless}"
if [ "$DOCKER_MODE" = rootful ]; then
  READINESS_DOCKER_HOST="${DOCKER_HOST:-unix:///var/run/docker.sock}"
else
  READINESS_XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/xdg-test-$USER}"
  READINESS_DOCKER_HOST="${DOCKER_HOST:-unix://$READINESS_XDG_RUNTIME_DIR/docker.sock}"
fi
SRUN_RETRIES="${SRUN_RETRIES:-5}"
SRUN_RETRY_DELAY_SEC="${SRUN_RETRY_DELAY_SEC:-2}"

APPLY_HARBOR_ROOTLESS_PATCH="${APPLY_HARBOR_ROOTLESS_PATCH:-true}"
CHECK_DATASETS="${CHECK_DATASETS:-true}"
CHECK_MODEL_PATHS="${CHECK_MODEL_PATHS:-true}"
CHECK_ROOTLESS_DOCKER_PREREQS="${CHECK_ROOTLESS_DOCKER_PREREQS:-true}"
CHECK_MODEL_DOWNLOAD_TOOL="${CHECK_MODEL_DOWNLOAD_TOOL:-true}"
CHECK_STORAGE_HEADROOM="${CHECK_STORAGE_HEADROOM:-true}"
CHECK_SOCKET_IFNAME="${CHECK_SOCKET_IFNAME:-true}"
RUN_ROLLOUT_CUDA_RUNTIME_SMOKE="${RUN_ROLLOUT_CUDA_RUNTIME_SMOKE:-true}"
START_ROOTLESS_DOCKER="${START_ROOTLESS_DOCKER:-false}"
RUN_HARBOR_ROOTLESS_SMOKE="${RUN_HARBOR_ROOTLESS_SMOKE:-false}"
RUN_HARBOR_DOCKER_CONCURRENCY_SMOKE="${RUN_HARBOR_DOCKER_CONCURRENCY_SMOKE:-false}"
HARBOR_CONCURRENCY_TRIAL_COUNT="${HARBOR_CONCURRENCY_TRIAL_COUNT:-64}"
HARBOR_CONCURRENCY_MAX_IN_FLIGHT="${HARBOR_CONCURRENCY_MAX_IN_FLIGHT:-64}"
HARBOR_CONCURRENCY_MAX_FAILURES="${HARBOR_CONCURRENCY_MAX_FAILURES:-0}"
HARBOR_CONCURRENCY_DISABLE_PROJECT_NETWORK="${HARBOR_CONCURRENCY_DISABLE_PROJECT_NETWORK:-false}"
START_RAY_CLUSTER="${START_RAY_CLUSTER:-false}"
RUN_CLUSTER_VALIDATION="${RUN_CLUSTER_VALIDATION:-false}"
START_ROLLOUT_SERVERS="${START_ROLLOUT_SERVERS:-false}"
RUN_TRAINING_SMOKE="${RUN_TRAINING_SMOKE:-false}"
READINESS_RAY_START_MODE="${READINESS_RAY_START_MODE:-block}"
MIN_DATA_FREE_GB="${MIN_DATA_FREE_GB:-100}"
MIN_HOME_FREE_GB="${MIN_HOME_FREE_GB:-20}"
MIN_SCRATCH_FREE_GB="${MIN_SCRATCH_FREE_GB:-200}"

DATA_DIR_DEFAULT="$(cd "$REPO_ROOT/.." && pwd)/data/harbor"
DATA_DIR="${DATA_DIR:-$DATA_DIR_DEFAULT}"
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-$DATA_DIR/CodeContests}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-$DATA_DIR/OpenThoughts-TB-dev}"
TRAIN_DATA="${TRAIN_DATA:-['$TRAIN_DATA_DIR']}"
EVAL_DATA="${EVAL_DATA:-['$EVAL_DATA_DIR']}"
STORAGE_ROOT="${STORAGE_ROOT:-/data/zy/models/$USER}"
MODEL_ROOT="${MODEL_ROOT:-$STORAGE_ROOT/models}"
MODEL_PATH="${MODEL_PATH:-$MODEL_ROOT/Qwen3-32B}"
ROLLOUT_PORT_A="${ROLLOUT_PORT_A:-18000}"
ROLLOUT_PORT_B="${ROLLOUT_PORT_B:-18001}"
ROLLOUT_STARTUP_TIMEOUT_SEC="${ROLLOUT_STARTUP_TIMEOUT_SEC:-900}"
ROLLOUT_RUN_NAME="${ROLLOUT_RUN_NAME:-codecontest-qwen3-32b-rollout-job${JOB_ID}-$(date +%s)}"

RAY_LAUNCH_PID=""
ROLLOUT_PID=""
ROOTLESS_DOCKER_PID=""
FAILURES=0
FAILED_CHECKS=()

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

run_on_node() {
  local node="$1"
  shift
  local attempt=1
  local rc=0
  local output_file=""
  while [ "$attempt" -le "$SRUN_RETRIES" ]; do
    output_file="$(mktemp)"
    if srun --jobid "$JOB_ID" --overlap --overcommit --immediate=10 -w "$node" --ntasks=1 --nodes=1 --cpus-per-task=1 --gres=gpu:0 bash -lc "$*" >"$output_file" 2>&1; then
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

wait_for_rollout_health() {
  local node="$1"
  local host="$2"
  local port_a="$3"
  local port_b="$4"
  local launcher_pid="${5:-}"
  local launcher_log="${6:-}"
  local deadline=$((SECONDS + ROLLOUT_STARTUP_TIMEOUT_SEC))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if [ -n "$launcher_log" ] && grep -q "External rollout servers ready:" "$launcher_log" 2>/dev/null; then
      if curl -sf "http://$host:$port_a/health" >/dev/null && curl -sf "http://$host:$port_b/health" >/dev/null; then
        echo "Rollout servers are healthy on $node"
        return 0
      fi
    fi
    if [ -n "$launcher_pid" ]; then
      local stat=""
      stat="$(ps -o stat= -p "$launcher_pid" 2>/dev/null | tr -d '[:space:]')"
      if [ -z "$stat" ] || [[ "$stat" == Z* ]]; then
        [ -n "$launcher_log" ] && cat "$launcher_log" >&2
        echo "Rollout launcher exited before reporting ready" >&2
        return 1
      fi
    fi
    sleep 5
  done
  [ -n "$launcher_log" ] && cat "$launcher_log" >&2
  echo "Timed out waiting for rollout servers on $node" >&2
  return 1
}

wait_for_rollout_node_health_probe() {
  local node="$1"
  local port_a="$2"
  local port_b="$3"
  local deadline=$((SECONDS + ROLLOUT_STARTUP_TIMEOUT_SEC))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if run_on_node "$node" "curl -sf 'http://127.0.0.1:$port_a/health' >/dev/null && curl -sf 'http://127.0.0.1:$port_b/health' >/dev/null"; then
      echo "Rollout servers are healthy on $node"
      return 0
    fi
    sleep 5
  done
  echo "Timed out waiting for rollout servers on $node" >&2
  return 1
}

wait_for_head_docker_ready() {
  local deadline=$((SECONDS + 60))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if run_on_node "$HEAD_NODE" "DOCKER_HOST='$READINESS_DOCKER_HOST' docker info >/dev/null"; then
      echo "Docker is healthy on $HEAD_NODE via $READINESS_DOCKER_HOST"
      return 0
    fi
    sleep 2
  done
  echo "Timed out waiting for Docker on $HEAD_NODE via $READINESS_DOCKER_HOST" >&2
  return 1
}

run_training_smoke_check() {
  local output_file=""
  local rc=0
  output_file="$(mktemp)"
  if run_on_node "$HEAD_NODE" "cd '$REPO_ROOT' && export RAY_HEAD_IP='$HEAD_IP' && export ROLLOUT_HOST_IP='$ROLLOUT_IP' && export DOCKER_MODE='$DOCKER_MODE' && export DOCKER_HOST='$READINESS_DOCKER_HOST' && export TRAIN_DATA='$TRAIN_DATA' && export EVAL_DATA='$EVAL_DATA' && bash '$SCRIPT_DIR/run_codecontest_qwen3_32b_6node_rootless_fully_async.sh' smoke && echo TRAINING_SMOKE_OK" >"$output_file" 2>&1; then
    cat "$output_file"
    if ! grep -q '^TRAINING_SMOKE_OK$' "$output_file"; then
      echo "Training smoke did not emit the success sentinel" >&2
      rm -f "$output_file"
      return 1
    fi
    rm -f "$output_file"
    return 0
  else
    rc=$?
  fi
  cat "$output_file" >&2
  rm -f "$output_file"
  return "$rc"
}

cleanup() {
  if [ -n "$ROLLOUT_PID" ]; then
    kill "$ROLLOUT_PID" 2>/dev/null || true
    wait "$ROLLOUT_PID" 2>/dev/null || true
  fi
  if [ -n "$ROOTLESS_DOCKER_PID" ]; then
    kill "$ROOTLESS_DOCKER_PID" 2>/dev/null || true
    wait "$ROOTLESS_DOCKER_PID" 2>/dev/null || true
  fi
  if [ -n "$RAY_LAUNCH_PID" ]; then
    kill "$RAY_LAUNCH_PID" 2>/dev/null || true
    wait "$RAY_LAUNCH_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

record_failure() {
  local name="$1"
  FAILURES=$((FAILURES + 1))
  FAILED_CHECKS+=("$name")
}

run_check() {
  local name="$1"
  shift
  echo
  echo "$name"
  set +e
  "$@"
  local rc=$?
  set -e
  if [ "$rc" -eq 0 ]; then
    echo "PASS: $name"
  else
    echo "FAIL: $name"
    record_failure "$name"
  fi
  return 0
}

run_shell_check() {
  local name="$1"
  shift
  run_check "$name" bash -lc "$*"
}

set -e

require_cmd squeue
require_cmd scontrol
require_cmd srun
require_env JOB_ID

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
if [ -z "$HEAD_NODE" ]; then
  HEAD_NODE="${JOB_NODES[0]}"
fi
if [ -z "$ROLLOUT_NODE" ] && [ "${#JOB_NODES[@]}" -ge 6 ]; then
  ROLLOUT_NODE="${JOB_NODES[-1]}"
fi
if [ -z "$TRAINER_NODES_CSV" ]; then
  TRAINER_NODES=()
  for node in "${JOB_NODES[@]}"; do
    if [ "$node" != "$HEAD_NODE" ] && [ "$node" != "$ROLLOUT_NODE" ]; then
      TRAINER_NODES+=("$node")
    fi
  done
  TRAINER_NODES_CSV="$(IFS=,; echo "${TRAINER_NODES[*]}")"
fi

IFS=',' read -r -a TRAINER_NODES <<<"$TRAINER_NODES_CSV"

HEAD_IP="$(node_ip "$HEAD_NODE")"
ROLLOUT_IP=""
if [ -n "$ROLLOUT_NODE" ]; then
  ROLLOUT_IP="$(node_ip "$ROLLOUT_NODE")"
fi

ALL_NODES=("$HEAD_NODE" "${TRAINER_NODES[@]}")
if [ -n "$ROLLOUT_NODE" ]; then
  ALL_NODES+=("$ROLLOUT_NODE")
fi

echo "Readiness topology"
echo "  job_id:        $JOB_ID"
echo "  head_node:     $HEAD_NODE ($HEAD_IP)"
echo "  trainer_nodes: $TRAINER_NODES_CSV"
echo "  rollout_node:  ${ROLLOUT_NODE:-unset} ${ROLLOUT_IP:+($ROLLOUT_IP)}"

if [ "$CHECK_MODEL_DOWNLOAD_TOOL" = true ]; then
  run_check \
    "Model download CLI" \
    run_on_node "$HEAD_NODE" "[ -x '$REPO_ROOT/.venv/bin/hf' ] || [ -x '$REPO_ROOT/.venv/bin/huggingface-cli' ] || command -v hf >/dev/null 2>&1 || command -v huggingface-cli >/dev/null 2>&1 || { echo 'Hugging Face download CLI missing'; exit 1; }"
fi

if [ "$CHECK_STORAGE_HEADROOM" = true ]; then
  for node in "${ALL_NODES[@]}"; do
    run_check \
      "Storage headroom on $node" \
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

if [ "$CHECK_SOCKET_IFNAME" = true ]; then
  echo "Checking per-node socket interface detection"
  for node in "${ALL_NODES[@]}"; do
    run_check \
      "Socket IFNAME on $node" \
      run_on_node "$node" "iface=\$(bash '$SCRIPT_DIR/detect_socket_ifname.sh'); echo socket_ifname=\$iface; [ -n \"\$iface\" ]"
  done
fi

if [ "$APPLY_HARBOR_ROOTLESS_PATCH" = true ]; then
  run_check \
    "Harbor rootless patch" \
    bash -lc "'$PYTHON_BIN' '$OPS_DIR/apply_harbor_runtime_patches.py' --backup && '$PYTHON_BIN' '$OPS_DIR/apply_harbor_runtime_patches.py' --check"
fi

if [ "$CHECK_DATASETS" = true ]; then
  run_check \
    "Dataset validation" \
    "$PYTHON_BIN" "$VALIDATION_DIR/validate_harbor_task_dataset.py" \
      --train-data "$TRAIN_DATA" \
      --eval-data "$EVAL_DATA" \
      --stage "$STAGE"
fi

if [ "$CHECK_MODEL_PATHS" = true ]; then
  run_check \
    "Model path on head" \
    run_on_node "$HEAD_NODE" "[ -f '$MODEL_PATH/config.json' ] || { echo 'model missing on head: $MODEL_PATH'; exit 1; }"
  if [ -n "$ROLLOUT_NODE" ]; then
    run_check \
      "Model path on rollout" \
      run_on_node "$ROLLOUT_NODE" "[ -f '$MODEL_PATH/config.json' ] || { echo 'model missing on rollout node: $MODEL_PATH'; exit 1; }"
  fi
fi

if [ "$RUN_ROLLOUT_CUDA_RUNTIME_SMOKE" = true ] && [ -n "$ROLLOUT_NODE" ]; then
  run_check \
    "Rollout CUDA runtime smoke" \
    bash -lc "JOB_ID='$JOB_ID' ROLLOUT_NODE='$ROLLOUT_NODE' PYTHON_BIN='$PYTHON_BIN' bash '$VALIDATION_DIR/check_rollout_cuda_runtime_smoke.sh'"
fi

if [ "$CHECK_ROOTLESS_DOCKER_PREREQS" = true ]; then
  run_check \
    "$([ "$DOCKER_MODE" = rootless ] && echo "Rootless Docker prerequisites" || echo "Docker prerequisites")" \
    bash -lc "run_on_node() { srun --jobid '$JOB_ID' --overlap --overcommit --immediate=10 -w '$HEAD_NODE' --ntasks=1 --nodes=1 --cpus-per-task=1 bash -lc \"\$*\"; }; run_on_node 'command -v docker >/dev/null 2>&1 || { echo docker missing; exit 1; }'; run_on_node 'docker compose version >/dev/null 2>&1 || { echo docker compose plugin missing; exit 1; }'; if [ '$DOCKER_MODE' = 'rootless' ]; then run_on_node 'command -v slirp4netns >/dev/null 2>&1 || { echo slirp4netns missing; exit 1; }'; run_on_node 'command -v rootlesskit >/dev/null 2>&1 || { echo rootlesskit missing; exit 1; }'; run_on_node 'command -v newuidmap >/dev/null 2>&1 || { echo newuidmap missing; exit 1; }'; run_on_node 'command -v newgidmap >/dev/null 2>&1 || { echo newgidmap missing; exit 1; }'; run_on_node 'command -v dockerd-rootless.sh >/dev/null 2>&1 || { echo dockerd-rootless.sh missing; exit 1; }'; fi"
fi

if [ "$START_ROOTLESS_DOCKER" = true ]; then
  ROOTLESS_DOCKER_STARTUP_LOG="$(mktemp)"
  echo
  echo "Starting rootless Docker on head"
  echo "  rootless_docker_startup_log: $ROOTLESS_DOCKER_STARTUP_LOG"
  srun --jobid "$JOB_ID" --overlap -w "$HEAD_NODE" --ntasks=1 --nodes=1 --cpus-per-task=1 --gres=gpu:0 bash -lc \
    "cd '$REPO_ROOT' && export DOCKER_HOST='$READINESS_DOCKER_HOST' && export XDG_RUNTIME_DIR='${READINESS_XDG_RUNTIME_DIR:-}' && export ROOTLESS_DOCKER_START_MODE=block && bash '$SCRIPT_DIR/start_rootless_docker_for_harbor.sh'" \
    >"$ROOTLESS_DOCKER_STARTUP_LOG" 2>&1 &
  ROOTLESS_DOCKER_PID="$!"
  run_check \
    "Start rootless Docker on head" \
    wait_for_head_docker_ready
fi

if [ "$RUN_HARBOR_ROOTLESS_SMOKE" = true ]; then
  run_check \
    "Harbor rootless verifier smoke" \
    run_on_node "$HEAD_NODE" "cd '$REPO_ROOT' && export DOCKER_HOST='$READINESS_DOCKER_HOST' && '$PYTHON_BIN' '$VALIDATION_DIR/harbor_rootless_verifier_smoke.py' --docker-host \"\$DOCKER_HOST\" --clean"
fi

if [ "$RUN_HARBOR_DOCKER_CONCURRENCY_SMOKE" = true ]; then
  run_check \
    "Harbor docker concurrency smoke" \
    bash -lc "JOB_ID='$JOB_ID' HEAD_NODE='$HEAD_NODE' DOCKER_MODE='$DOCKER_MODE' TRIAL_COUNT='$HARBOR_CONCURRENCY_TRIAL_COUNT' MAX_CONCURRENCY='$HARBOR_CONCURRENCY_MAX_IN_FLIGHT' MAX_FAILURES='$HARBOR_CONCURRENCY_MAX_FAILURES' DISABLE_PROJECT_NETWORK='$HARBOR_CONCURRENCY_DISABLE_PROJECT_NETWORK' PYTHON_BIN='$PYTHON_BIN' OUTPUT_ROOT='$LOG_ROOT/harbor-docker-concurrency-smoke-job${JOB_ID}' bash '$VALIDATION_DIR/run_harbor_docker_concurrency_smoke.sh'"
fi

if [ "$START_ROLLOUT_SERVERS" = true ]; then
  require_env ROLLOUT_NODE
  ROLLOUT_STARTUP_LOG="$(mktemp)"
  echo
  echo "Starting rollout servers"
  echo "  rollout_startup_log: $ROLLOUT_STARTUP_LOG"
  srun --jobid "$JOB_ID" --overlap -w "$ROLLOUT_NODE" --ntasks=1 --nodes=1 --cpus-per-task=100 --gres=gpu:8 bash -lc \
    "cd '$REPO_ROOT' && export RAY_HEAD_IP='$HEAD_IP' && export RUN_NAME='$ROLLOUT_RUN_NAME' && export PORT_A='$ROLLOUT_PORT_A' && export PORT_B='$ROLLOUT_PORT_B' && bash '$SCRIPT_DIR/start_qwen3_32b_external_rollout_servers.sh'" \
    >"$ROLLOUT_STARTUP_LOG" 2>&1 &
  ROLLOUT_PID="$!"
  run_check \
    "Rollout startup" \
    wait_for_rollout_health "$ROLLOUT_NODE" "$ROLLOUT_IP" "$ROLLOUT_PORT_A" "$ROLLOUT_PORT_B" "$ROLLOUT_PID" "$ROLLOUT_STARTUP_LOG"
fi

if [ "$START_RAY_CLUSTER" = true ]; then
  echo
  echo "Starting Ray cluster"
  JOB_ID="$JOB_ID" \
  HEAD_NODE="$HEAD_NODE" \
  TRAINER_NODES_CSV="$TRAINER_NODES_CSV" \
  RAY_START_MODE="$READINESS_RAY_START_MODE" \
  bash "$SCRIPT_DIR/launch_qwen3_32b_ray_cluster.sh" &
  RAY_LAUNCH_PID="$!"
  if [ "$READINESS_RAY_START_MODE" = detached ]; then
    wait "$RAY_LAUNCH_PID"
    RAY_LAUNCH_PID=""
  else
    sleep 5
  fi
fi

if [ "$RUN_CLUSTER_VALIDATION" = true ]; then
  VALIDATION_DOCKER_HOST="$READINESS_DOCKER_HOST"
  VALIDATION_SKIP_ROLLOUT_CUDA_RUNTIME_SMOKE=false
  if [ "$START_ROLLOUT_SERVERS" = true ] || [ "$RUN_ROLLOUT_CUDA_RUNTIME_SMOKE" = true ]; then
    VALIDATION_SKIP_ROLLOUT_CUDA_RUNTIME_SMOKE=true
  fi
  run_check \
    "Cluster validation" \
    bash -lc "JOB_ID='$JOB_ID' HEAD_NODE='$HEAD_NODE' TRAINER_NODES_CSV='$TRAINER_NODES_CSV' ROLLOUT_NODE='$ROLLOUT_NODE' VALIDATION_MODE='$([ "$START_ROLLOUT_SERVERS" = true ] && echo full || echo train-cluster-only)' DOCKER_MODE='$DOCKER_MODE' DOCKER_HOST='$VALIDATION_DOCKER_HOST' SKIP_ROLLOUT_CUDA_RUNTIME_SMOKE='$VALIDATION_SKIP_ROLLOUT_CUDA_RUNTIME_SMOKE' TRAIN_DATA='$TRAIN_DATA' EVAL_DATA='$EVAL_DATA' bash '$SCRIPT_DIR/run_full_run_validation_suite.sh'"
fi

if [ "$RUN_TRAINING_SMOKE" = true ]; then
  if [ -z "$ROLLOUT_NODE" ]; then
    record_failure "Training smoke"
    echo
    echo "FAIL: Training smoke"
    echo "ROLLOUT_NODE is required"
  else
    run_check \
      "Training smoke" \
      run_training_smoke_check
  fi
fi

echo
echo "Readiness suite completed"
echo "  failures: $FAILURES"
if [ "$FAILURES" -gt 0 ]; then
  printf '  failed_checks: %s\n' "${FAILED_CHECKS[*]}"
  exit 1
fi
