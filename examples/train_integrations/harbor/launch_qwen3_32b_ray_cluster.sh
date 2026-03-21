#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
RAY_BIN="${RAY_BIN:-$REPO_ROOT/.venv/bin/ray}"

JOB_ID="${JOB_ID:-}"
RAY_START_MODE="${RAY_START_MODE:-block}"
RAY_PORT="${RAY_PORT:-6381}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8265}"
SRUN_RETRIES="${SRUN_RETRIES:-5}"
SRUN_RETRY_DELAY_SEC="${SRUN_RETRY_DELAY_SEC:-2}"
HEAD_NODE="${HEAD_NODE:-}"
ROLLOUT_NODE="${ROLLOUT_NODE:-}"
TRAINER_NODES_CSV="${TRAINER_NODES_CSV:-}"
HEAD_CPUS="${HEAD_CPUS:-8}"
HEAD_GPUS="${HEAD_GPUS:-0}"
TRAINER_CPUS="${TRAINER_CPUS:-176}"
TRAINER_GPUS="${TRAINER_GPUS:-8}"
HEAD_NOFILE_SOFT="${HEAD_NOFILE_SOFT:-}"
TRAINER_NOFILE_SOFT="${TRAINER_NOFILE_SOFT:-$HEAD_NOFILE_SOFT}"
HEAD_METRICS_PORT="${HEAD_METRICS_PORT:-28080}"
TRAINER_METRICS_PORT_BASE="${TRAINER_METRICS_PORT_BASE:-28100}"
RAY_MIN_WORKER_PORT="${RAY_MIN_WORKER_PORT:-30000}"
RAY_MAX_WORKER_PORT="${RAY_MAX_WORKER_PORT:-30999}"
HEAD_DASHBOARD_AGENT_LISTEN_PORT="${HEAD_DASHBOARD_AGENT_LISTEN_PORT:-28280}"
HEAD_DASHBOARD_AGENT_GRPC_PORT="${HEAD_DASHBOARD_AGENT_GRPC_PORT:-28380}"
HEAD_RUNTIME_ENV_AGENT_PORT="${HEAD_RUNTIME_ENV_AGENT_PORT:-28480}"
TRAINER_DASHBOARD_AGENT_LISTEN_PORT_BASE="${TRAINER_DASHBOARD_AGENT_LISTEN_PORT_BASE:-28200}"
TRAINER_DASHBOARD_AGENT_GRPC_PORT_BASE="${TRAINER_DASHBOARD_AGENT_GRPC_PORT_BASE:-28300}"
TRAINER_RUNTIME_ENV_AGENT_PORT_BASE="${TRAINER_RUNTIME_ENV_AGENT_PORT_BASE:-28400}"
STARTUP_TIMEOUT_SEC="${STARTUP_TIMEOUT_SEC:-60}"

SRUN_PIDS=()
EXPECTED_TOTAL_CPUS=0
EXPECTED_TOTAL_GPUS=0

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
    if srun --jobid "$JOB_ID" --overlap --overcommit --exact --gpus-per-node=0 --immediate=10 -w "$node" --ntasks=1 --nodes=1 --cpus-per-task=1 bash -lc "$*" >"$output_file" 2>&1; then
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

run_on_node_bg() {
  local node="$1"
  shift
  srun --jobid "$JOB_ID" --overlap --overcommit --exact --gpus-per-node=0 --immediate=10 -w "$node" --ntasks=1 --nodes=1 --cpus-per-task=1 bash -lc "$*" &
  SRUN_PIDS+=("$!")
}

node_ip() {
  local node="$1"
  run_on_node "$node" "hostname -I | awk '{print \$1}'"
}

build_nofile_snippet() {
  local requested_soft="$1"
  if [ -z "$requested_soft" ]; then
    return 0
  fi
  if ! [[ "$requested_soft" =~ ^[0-9]+$ ]]; then
    echo "Invalid nofile soft limit: $requested_soft" >&2
    exit 1
  fi
  cat <<EOF
NOFILE_SOFT='$requested_soft' && CURRENT_SOFT=\$(ulimit -Sn) && CURRENT_HARD=\$(ulimit -Hn) && if [ "\$CURRENT_SOFT" != "unlimited" ] && [ "\$CURRENT_SOFT" -lt "\$NOFILE_SOFT" ]; then if ! ulimit -Sn "\$NOFILE_SOFT" 2>/dev/null; then echo "Failed to raise soft nofile from \$CURRENT_SOFT to \$NOFILE_SOFT (hard=\$CURRENT_HARD)." >&2; exit 1; fi; fi &&
EOF
}

start_head() {
  local node="$1"
  local block_flag=""
  if [ "$RAY_START_MODE" = block ]; then
    block_flag=" --block"
  fi
  local nofile_snippet=""
  nofile_snippet="$(build_nofile_snippet "$HEAD_NOFILE_SOFT")"
  local cmd="cd '$REPO_ROOT' && ${nofile_snippet}'$RAY_BIN' stop -f >/tmp/ray_stop_head.log 2>&1 || true && NODE_IP=\$(hostname -I | awk '{print \$1}') && HARBOR_HEAD_RESOURCES=\$('${PYTHON_BIN}' -c \"import json; print(json.dumps({'harbor_head': 1}))\") && '$RAY_BIN' start --head --disable-usage-stats${block_flag} --port '$RAY_PORT' --dashboard-host 0.0.0.0 --dashboard-port '$DASHBOARD_PORT' --dashboard-agent-listen-port '$HEAD_DASHBOARD_AGENT_LISTEN_PORT' --dashboard-agent-grpc-port '$HEAD_DASHBOARD_AGENT_GRPC_PORT' --runtime-env-agent-port '$HEAD_RUNTIME_ENV_AGENT_PORT' --min-worker-port '$RAY_MIN_WORKER_PORT' --max-worker-port '$RAY_MAX_WORKER_PORT' --node-ip-address \"\$NODE_IP\" --num-cpus '$HEAD_CPUS' --num-gpus '$HEAD_GPUS' --metrics-export-port '$HEAD_METRICS_PORT' --resources \"\$HARBOR_HEAD_RESOURCES\""
  if [ "$RAY_START_MODE" = block ]; then
    run_on_node_bg "$node" "$cmd"
  else
    run_on_node "$node" "$cmd"
  fi
}

start_trainer() {
  local node="$1"
  local head_ip="$2"
  local metrics_port="$3"
  local dashboard_agent_listen_port="$4"
  local dashboard_agent_grpc_port="$5"
  local runtime_env_agent_port="$6"
  local block_flag=""
  if [ "$RAY_START_MODE" = block ]; then
    block_flag=" --block"
  fi
  local nofile_snippet=""
  nofile_snippet="$(build_nofile_snippet "$TRAINER_NOFILE_SOFT")"
  local cmd="cd '$REPO_ROOT' && ${nofile_snippet}'$RAY_BIN' stop -f >/tmp/ray_stop_worker.log 2>&1 || true && NODE_IP=\$(hostname -I | awk '{print \$1}') && '$RAY_BIN' start --disable-usage-stats${block_flag} --address '$head_ip:$RAY_PORT' --dashboard-agent-listen-port '$dashboard_agent_listen_port' --dashboard-agent-grpc-port '$dashboard_agent_grpc_port' --runtime-env-agent-port '$runtime_env_agent_port' --min-worker-port '$RAY_MIN_WORKER_PORT' --max-worker-port '$RAY_MAX_WORKER_PORT' --node-ip-address \"\$NODE_IP\" --num-cpus '$TRAINER_CPUS' --num-gpus '$TRAINER_GPUS' --metrics-export-port '$metrics_port'"
  if [ "$RAY_START_MODE" = block ]; then
    run_on_node_bg "$node" "$cmd"
  else
    run_on_node "$node" "$cmd"
  fi
}

cleanup() {
  if [ "${#SRUN_PIDS[@]}" -eq 0 ]; then
    return 0
  fi
  kill "${SRUN_PIDS[@]}" 2>/dev/null || true
  wait "${SRUN_PIDS[@]}" 2>/dev/null || true
}

wait_for_cluster() {
  local ray_address="$1"
  local deadline=$((SECONDS + STARTUP_TIMEOUT_SEC))
  local status_file
  status_file="$(mktemp)"
  while true; do
    if "$RAY_BIN" status --address "$ray_address" >"$status_file" 2>&1 \
      && ! grep -q "No cluster status" "$status_file" \
      && grep -Fq "/${EXPECTED_TOTAL_CPUS}.0 CPU" "$status_file" \
      && grep -Fq "/${EXPECTED_TOTAL_GPUS}.0 GPU" "$status_file" \
      && grep -Fq "/1.0 harbor_head" "$status_file"; then
      cat "$status_file"
      rm -f "$status_file"
      return 0
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
      cat "$status_file" >&2
      rm -f "$status_file"
      echo "Timed out waiting for Ray cluster at $ray_address" >&2
      return 1
    fi
    sleep 2
  done
}

require_cmd squeue
require_cmd scontrol
require_cmd srun
require_env JOB_ID

case "$RAY_START_MODE" in
  block|detached)
    ;;
  *)
    echo "Unsupported RAY_START_MODE: $RAY_START_MODE" >&2
    exit 1
    ;;
esac

trap cleanup EXIT INT TERM

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python env not found: $PYTHON_BIN" >&2
  exit 1
fi

if [ ! -x "$RAY_BIN" ]; then
  echo "Ray CLI not found: $RAY_BIN" >&2
  exit 1
fi

NODELIST="$(squeue -j "$JOB_ID" -h -o '%N')"
if [ -z "$NODELIST" ]; then
  echo "No active nodes found for job $JOB_ID" >&2
  exit 1
fi

mapfile -t JOB_NODES < <(scontrol show hostnames "$NODELIST")
if [ "${#JOB_NODES[@]}" -lt 5 ]; then
  echo "Expected at least 5 allocated nodes, got ${#JOB_NODES[@]} node(s): ${JOB_NODES[*]}" >&2
  exit 1
fi

if [ -z "$HEAD_NODE" ]; then
  HEAD_NODE="${JOB_NODES[0]}"
fi

if [ -n "$TRAINER_NODES_CSV" ]; then
  IFS=',' read -r -a TRAINER_NODES <<<"$TRAINER_NODES_CSV"
else
  TRAINER_NODES=()
  for node in "${JOB_NODES[@]}"; do
    if [ "$node" != "$HEAD_NODE" ] && [ -z "$ROLLOUT_NODE" -o "$node" != "$ROLLOUT_NODE" ]; then
      TRAINER_NODES+=("$node")
    fi
  done
fi

if [ "${#TRAINER_NODES[@]}" -ne 4 ]; then
  echo "Expected exactly 4 trainer nodes, got ${#TRAINER_NODES[@]}: ${TRAINER_NODES[*]}" >&2
  exit 1
fi

echo "Launching Ray cluster"
echo "  job_id:        $JOB_ID"
echo "  head_node:     $HEAD_NODE"
echo "  trainer_nodes: ${TRAINER_NODES[*]}"
echo "  ray_port:      $RAY_PORT"
echo "  start_mode:    $RAY_START_MODE"

EXPECTED_TOTAL_CPUS=$((HEAD_CPUS + TRAINER_CPUS * ${#TRAINER_NODES[@]}))
EXPECTED_TOTAL_GPUS=$((TRAINER_GPUS * ${#TRAINER_NODES[@]}))

HEAD_IP="$(node_ip "$HEAD_NODE")"
start_head "$HEAD_NODE"

for i in "${!TRAINER_NODES[@]}"; do
  metrics_port=$((TRAINER_METRICS_PORT_BASE + i))
  dashboard_agent_listen_port=$((TRAINER_DASHBOARD_AGENT_LISTEN_PORT_BASE + i))
  dashboard_agent_grpc_port=$((TRAINER_DASHBOARD_AGENT_GRPC_PORT_BASE + i))
  runtime_env_agent_port=$((TRAINER_RUNTIME_ENV_AGENT_PORT_BASE + i))
  start_trainer "${TRAINER_NODES[$i]}" "$HEAD_IP" "$metrics_port" "$dashboard_agent_listen_port" "$dashboard_agent_grpc_port" "$runtime_env_agent_port"
done

echo
echo "Verifying Ray cluster"
wait_for_cluster "${HEAD_IP}:${RAY_PORT}"

echo
echo "Ray cluster ready at ${HEAD_IP}:${RAY_PORT}"
if [ "$RAY_START_MODE" = block ]; then
  echo "Blocking mode is active; keep this process alive while running validation or training."
  wait
fi
