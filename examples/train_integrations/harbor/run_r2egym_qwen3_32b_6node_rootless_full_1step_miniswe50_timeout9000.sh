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
HEAD_IP="${HEAD_IP:-}"
ROLLOUT_IP="${ROLLOUT_IP:-}"
RAY_PORT="${RAY_PORT:-6381}"
DOCKER_MODE="${DOCKER_MODE:-rootless}"
ROLLOUT_SERVER_PORTS_CSV="${ROLLOUT_SERVER_PORTS_CSV:-18000,18001,18002,18003}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-2}"
ROLLOUT_ENGINES="${ROLLOUT_ENGINES:-4}"
ROLLOUT_NOFILE_SOFT="${ROLLOUT_NOFILE_SOFT:-131072}"
HEAD_NOFILE_SOFT="${HEAD_NOFILE_SOFT:-131072}"
TRAINER_NOFILE_SOFT="${TRAINER_NOFILE_SOFT:-131072}"
DOCKER_NOFILE_SOFT="${DOCKER_NOFILE_SOFT:-131072}"
DOCKER_INOTIFY_MAX_USER_INSTANCES="${DOCKER_INOTIFY_MAX_USER_INSTANCES:-}"
HEAD_DOCKER_READY_TIMEOUT_SEC="${HEAD_DOCKER_READY_TIMEOUT_SEC:-300}"
AGENT_TIMEOUT_SEC="${AGENT_TIMEOUT_SEC:-9000}"
MINI_SWE_MODEL_TIMEOUT_SEC="${MINI_SWE_MODEL_TIMEOUT_SEC:-1200}"
MAX_TRAIN_TASKS="${MAX_TRAIN_TASKS:-64}"
MONITOR_INTERVAL_SEC="${MONITOR_INTERVAL_SEC:-10}"
THUNDERAGENT_WATCHDOG_INTERVAL_SEC="${THUNDERAGENT_WATCHDOG_INTERVAL_SEC:-15}"
THUNDERAGENT_WATCHDOG_FAILURES_BEFORE_DUMP="${THUNDERAGENT_WATCHDOG_FAILURES_BEFORE_DUMP:-3}"
THUNDERAGENT_WATCHDOG_ENABLED="${THUNDERAGENT_WATCHDOG_ENABLED:-1}"
TRAIN_DATA="${TRAIN_DATA:-['/home/hkang/zthunder_agent/data/harbor/r2egym-trivial','/home/hkang/zthunder_agent/data/harbor/r2egym-easy','/home/hkang/zthunder_agent/data/harbor/r2egym-medium','/home/hkang/zthunder_agent/data/harbor/r2egym-hard']}"
EVAL_DATA="${EVAL_DATA:-$TRAIN_DATA}"

RUN_NAME="${RUN_NAME_OVERRIDE:-r2egym-qwen3-32b-6node-rootless-full-1step-4srv-miniswe50-timeout9000-$(date +%Y%m%d_%H%M%S)}"
RUN_SHORT_ID="${RUN_SHORT_ID:-r2e1s-$(date +%m%d%H%M%S)}"
LOG_DIR="${LOG_DIR:-$WORKSPACE_ROOT/tmp_logs/$RUN_NAME}"
RUN_ARTIFACT_ROOT="${RUN_ARTIFACT_ROOT:-$WORKSPACE_ROOT}"
RUN_ARTIFACT_DIR="$RUN_ARTIFACT_ROOT/$RUN_NAME"
TRAINER_MONITOR_ROOT="$LOG_DIR/trainer_monitors"
ROLLOUT_LOG_DIR="$LOG_DIR/rollout"
ROLLOUT_MONITOR_DIR="$ROLLOUT_LOG_DIR/monitoring"
ANALYSIS_DIR="$LOG_DIR/analysis"
SUMMARY_DIR="$LOG_DIR/post_run_summary"
TRAIN_DRIVER_LOG="$LOG_DIR/launcher_train_driver.log"
RAY_LOG="$LOG_DIR/launcher_ray.log"
ROLLOUT_LOG="$LOG_DIR/launcher_rollout.log"
TRAINER_MONITOR_LOG="$LOG_DIR/launcher_trainer_monitors.log"
THUNDERAGENT_WATCHDOG_LOG="$LOG_DIR/launcher_thunderagent_watchdog.log"
ROOTLESS_DOCKER_LOG="$LOG_DIR/launcher_rootless_docker.log"
SUMMARY_LOG="$LOG_DIR/launcher_summary.log"
ANALYSIS_LOG="$LOG_DIR/launcher_analysis.log"
SKYRL_WORKER_PID_HELPER="$SCRIPT_DIR/find_latest_skyrl_worker_pid.py"

ROOTLESS_SCRATCH_ROOT="${ROOTLESS_SCRATCH_ROOT:-/tmp/$USER/$RUN_SHORT_ID-rootless-scratch}"
TRAIN_RUNTIME_SCRATCH_ROOT="${TRAIN_RUNTIME_SCRATCH_ROOT:-/tmp/$USER/$RUN_SHORT_ID-train-runtime}"
ROLLOUT_RUNTIME_SCRATCH_ROOT="${ROLLOUT_RUNTIME_SCRATCH_ROOT:-/tmp/$USER/$RUN_SHORT_ID-rollout-runtime}"
XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/xdg-$RUN_SHORT_ID}"
DOCKER_HOST="${DOCKER_HOST:-unix://$XDG_RUNTIME_DIR/docker.sock}"
DOCKER_PIDFILE="${DOCKER_PIDFILE:-$XDG_RUNTIME_DIR/docker.pid}"
DOCKER_EXEC_ROOT="${DOCKER_EXEC_ROOT:-/tmp/$USER/$RUN_SHORT_ID-rootless-exec}"
DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT:-/scratch/triton_cache/$USER/r2egym-rootless-image-cache}"
DOCKER_LOG_PATH="${DOCKER_LOG_PATH:-$LOG_DIR/dockerd_rootless_harbor.log}"
HARBOR_SHARED_UV_CACHE_HOST_DIR="${HARBOR_SHARED_UV_CACHE_HOST_DIR:-/scratch/triton_cache/$USER/harbor-uv-cache}"
HARBOR_SHARED_UV_CACHE_ENV_DIR="${HARBOR_SHARED_UV_CACHE_ENV_DIR:-/harbor-shared/uv-cache}"
HARBOR_SHARED_MINI_SWE_TOOL_HOST_HOME="${HARBOR_SHARED_MINI_SWE_TOOL_HOST_HOME:-/scratch/triton_cache/$USER/harbor-mini-swe-home}"
HARBOR_SHARED_MINI_SWE_TOOL_ENV_HOME="${HARBOR_SHARED_MINI_SWE_TOOL_ENV_HOME:-$HARBOR_SHARED_MINI_SWE_TOOL_HOST_HOME}"
HARBOR_SHARED_UV_PYTHON_HOST_DIR="${HARBOR_SHARED_UV_PYTHON_HOST_DIR:-$WORKSPACE_ROOT/.local/share/uv/python}"
HARBOR_SHARED_UV_PYTHON_ENV_DIR="${HARBOR_SHARED_UV_PYTHON_ENV_DIR:-$HARBOR_SHARED_UV_PYTHON_HOST_DIR}"
HARBOR_MINI_SWE_AGENT_GIT_REF="${HARBOR_MINI_SWE_AGENT_GIT_REF:-8e8a515fdcecf3a8e45c3909f7f196bfe18ca89a}"
HARBOR_MINI_SWE_AGENT_UV_OFFLINE="${HARBOR_MINI_SWE_AGENT_UV_OFFLINE:-1}"
HARBOR_AGENT_MAX_TURNS="${HARBOR_AGENT_MAX_TURNS:-20}"

PIDS=()
INITIAL_STEP_IDS=()
STEPS_CLEANED_UP=false
BACKGROUND_CLEANED_UP=false

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

trim_csv_array() {
  local -n ref="$1"
  local idx=""
  for idx in "${!ref[@]}"; do
    ref[$idx]="$(printf '%s' "${ref[$idx]}" | xargs)"
  done
}

node_ip() {
  local node="$1"
  srun --jobid "$JOB_ID" --overlap --overcommit --immediate=10 -w "$node" --ntasks=1 --nodes=1 --cpus-per-task=1 --gres=gpu:0 \
    bash -lc "hostname -I | awk '{print \$1}'"
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
  local idx=""
  local port=""
  local suffixes=(a b c d e f g h i j k l m n o p)
  for idx in "$@"; do :; done
  idx=0
  for port in "$@"; do
    specs+=("rollout_${suffixes[$idx]}.log=http://${host_ip}:${port}")
    idx=$((idx + 1))
  done
  printf '%s\n' "$(IFS=';'; echo "${specs[*]}")"
}

collect_child_step_ids() {
  squeue -s -j "$JOB_ID" -h -o '%i' | grep -v '\.batch$' || true
}

cleanup_background() {
  if [ "$BACKGROUND_CLEANED_UP" = true ]; then
    return 0
  fi
  BACKGROUND_CLEANED_UP=true
  if [ "${#PIDS[@]}" -gt 0 ]; then
    kill "${PIDS[@]}" 2>/dev/null || true
    wait "${PIDS[@]}" 2>/dev/null || true
  fi
}

cleanup_new_steps() {
  if [ "$STEPS_CLEANED_UP" = true ]; then
    return 0
  fi
  STEPS_CLEANED_UP=true
  mapfile -t current_step_ids < <(collect_child_step_ids)
  if [ "${#current_step_ids[@]}" -eq 0 ]; then
    return 0
  fi
  local step_id=""
  local known=""
  local keep=false
  local -a new_step_ids=()
  for step_id in "${current_step_ids[@]}"; do
    keep=false
    for known in "${INITIAL_STEP_IDS[@]}"; do
      if [ "$step_id" = "$known" ]; then
        keep=true
        break
      fi
    done
    if [ "$keep" = false ]; then
      new_step_ids+=("$step_id")
    fi
  done
  if [ "${#new_step_ids[@]}" -gt 0 ]; then
    scancel "${new_step_ids[@]}" || true
  fi
}

final_cleanup() {
  cleanup_background
  cleanup_new_steps
}

prepare_shared_mini_swe_agent_tool_home() {
  mkdir -p "$HARBOR_SHARED_UV_CACHE_HOST_DIR" "$HARBOR_SHARED_MINI_SWE_TOOL_HOST_HOME"
  srun --jobid "$JOB_ID" --overlap --overcommit --immediate=10 -w "$HEAD_NODE" --ntasks=1 --nodes=1 --cpus-per-task=1 --gres=gpu:0 \
    bash -lc "set -euo pipefail
      export PATH='$REPO_ROOT/.venv/bin':\$HOME/.local/bin:\$PATH
      export HOME='$HARBOR_SHARED_MINI_SWE_TOOL_HOST_HOME'
      export UV_CACHE_DIR='$HARBOR_SHARED_UV_CACHE_HOST_DIR'
      mkdir -p '$HARBOR_SHARED_MINI_SWE_TOOL_HOST_HOME' '$HARBOR_SHARED_UV_CACHE_HOST_DIR'
      uv tool install --offline --cache-dir '$HARBOR_SHARED_UV_CACHE_HOST_DIR' git+https://github.com/li-boxuan/mini-swe-agent.git@'$HARBOR_MINI_SWE_AGENT_GIT_REF' >/tmp/mini-swe-agent-tool-home.log 2>&1 || \
      uv tool install --cache-dir '$HARBOR_SHARED_UV_CACHE_HOST_DIR' git+https://github.com/li-boxuan/mini-swe-agent.git@'$HARBOR_MINI_SWE_AGENT_GIT_REF' >>/tmp/mini-swe-agent-tool-home.log 2>&1 || {
        cat /tmp/mini-swe-agent-tool-home.log >&2
        exit 1
      }"
}

wait_for_head_docker() {
  local timeout_sec="${1:-120}"
  local deadline=$((SECONDS + timeout_sec))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if srun --jobid "$JOB_ID" --overlap --overcommit --immediate=10 -w "$HEAD_NODE" --ntasks=1 --nodes=1 --cpus-per-task=1 --gres=gpu:0 \
      bash -lc "export PATH='$REPO_ROOT/.venv/bin':\$HOME/.local/bin:\$PATH && export XDG_RUNTIME_DIR='$XDG_RUNTIME_DIR' && export DOCKER_HOST='$DOCKER_HOST' && timeout 10 docker info >/dev/null 2>&1"; then
      return 0
    fi
    sleep 2
  done
  echo "Timed out waiting for Docker at $DOCKER_HOST on $HEAD_NODE" >&2
  return 1
}

wait_for_ray() {
  local timeout_sec="${1:-180}"
  local deadline=$((SECONDS + timeout_sec))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if "$RAY_BIN" status --address "$HEAD_IP:$RAY_PORT" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "Timed out waiting for Ray at $HEAD_IP:$RAY_PORT" >&2
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

start_thunderagent_watchdog() {
  srun --jobid "$JOB_ID" --overlap --overcommit --immediate=10 -w "$HEAD_NODE" --ntasks=1 --nodes=1 --cpus-per-task=1 --gres=gpu:0 \
    bash -lc "set -euo pipefail
      target_url='http://$HEAD_IP:8080/health'
      interval_sec='$THUNDERAGENT_WATCHDOG_INTERVAL_SEC'
      failure_threshold='$THUNDERAGENT_WATCHDOG_FAILURES_BEFORE_DUMP'
      log_path='$THUNDERAGENT_WATCHDOG_LOG'
      python_bin='$PYTHON_BIN'
      pid_helper='$SKYRL_WORKER_PID_HELPER'
      ray_log_dir='/tmp/ray/session_latest/logs'
      failures=0
      armed=0
      last_dump_epoch=0
      while true; do
        if curl -sf --max-time 5 \"\$target_url\" >/dev/null 2>&1; then
          if [ \"\$armed\" -eq 0 ]; then
            echo \"\$(date '+%Y-%m-%d %H:%M:%S') health_ok arm_watchdog url=\$target_url\" >>\"\$log_path\"
            armed=1
          fi
          failures=0
        else
          if [ \"\$armed\" -eq 0 ]; then
            echo \"\$(date '+%Y-%m-%d %H:%M:%S') health_timeout_before_arm url=\$target_url\" >>\"\$log_path\"
            sleep \"\$interval_sec\"
            continue
          fi
          failures=\$((failures + 1))
          echo \"\$(date '+%Y-%m-%d %H:%M:%S') health_timeout failures=\$failures url=\$target_url\" >>\"\$log_path\"
          if [ \"\$failures\" -ge \"\$failure_threshold\" ]; then
            now_epoch=\$(date +%s)
            if [ \$((now_epoch - last_dump_epoch)) -ge 60 ]; then
              pid=\$(\"\$python_bin\" \"\$pid_helper\" --ray-log-dir \"\$ray_log_dir\" --pid-only 2>/dev/null || true)
              if [ -n \"\${pid:-}\" ] && kill -0 \"\$pid\" 2>/dev/null; then
                echo \"\$(date '+%Y-%m-%d %H:%M:%S') sending_SIGUSR1 pid=\$pid\" >>\"\$log_path\"
                kill -USR1 \"\$pid\" || true
                sleep 2
                ps -L -p \"\$pid\" -o pid,tid,pcpu,psr,stat,wchan:32,comm >>\"\$log_path\" 2>&1 || true
              else
                echo \"\$(date '+%Y-%m-%d %H:%M:%S') live_skyrl_entrypoint_pid_not_found\" >>\"\$log_path\"
              fi
              echo \"\$(date '+%Y-%m-%d %H:%M:%S') --- raylet tail ---\" >>\"\$log_path\"
              tail -n 80 \"\$ray_log_dir/raylet.out\" >>\"\$log_path\" 2>&1 || true
              echo \"\$(date '+%Y-%m-%d %H:%M:%S') --- gcs tail ---\" >>\"\$log_path\"
              tail -n 80 \"\$ray_log_dir/gcs_server.out\" >>\"\$log_path\" 2>&1 || true
              last_dump_epoch=\$now_epoch
            fi
          fi
        fi
        sleep \"\$interval_sec\"
      done" \
    >"$THUNDERAGENT_WATCHDOG_LOG" 2>&1 &
  PIDS+=("$!")
}

run_analysis() {
  mkdir -p "$ANALYSIS_DIR" "$SUMMARY_DIR"

  "$PYTHON_BIN" "$SCRIPT_DIR/summarize_qwen3_32b_full_run.py" \
    --run-name "$RUN_NAME" \
    --log-dir "$LOG_DIR" \
    --trainer-monitor-root "$TRAINER_MONITOR_ROOT" \
    --rollout-monitor-dir "$ROLLOUT_MONITOR_DIR" \
    --tensorboard-dir "$LOG_DIR/tensorboard" \
    --async-trace "$LOG_DIR/monitoring/async_trace.jsonl" \
    --output-dir "$SUMMARY_DIR" \
    --python-bin "$PYTHON_BIN" \
    >"$SUMMARY_LOG" 2>&1 || true

  {
    "$PYTHON_BIN" "$SCRIPT_DIR/analyze_thunderagent_monitoring.py" \
      --log-dir "$LOG_DIR" \
      --output-dir "$ANALYSIS_DIR"

    "$PYTHON_BIN" "$SCRIPT_DIR/analyze_qwen3_full_run_timeline.py" \
      --run-name "$RUN_NAME" \
      --log-root "$WORKSPACE_ROOT/tmp_logs" \
      --output-dir "$ANALYSIS_DIR"

    "$PYTHON_BIN" "$SCRIPT_DIR/analyze_harbor_step_timing.py" \
      --run-dir "$RUN_ARTIFACT_DIR" \
      --output-dir "$ANALYSIS_DIR" \
      --max-steps 50
  } >"$ANALYSIS_LOG" 2>&1 || true
}

trap final_cleanup EXIT INT TERM

require_cmd srun
require_cmd squeue
require_cmd scancel
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

mkdir -p "$LOG_DIR" "$TRAINER_MONITOR_ROOT" "$ROLLOUT_LOG_DIR" "$ANALYSIS_DIR" "$SUMMARY_DIR" "$RUN_ARTIFACT_DIR"

mapfile -t INITIAL_STEP_IDS < <(collect_child_step_ids)

IFS=',' read -r -a ROLLOUT_PORTS <<<"$ROLLOUT_SERVER_PORTS_CSV"
trim_csv_array ROLLOUT_PORTS

if [ -z "$HEAD_IP" ]; then
  HEAD_IP="$(node_ip "$HEAD_NODE")"
fi
if [ -z "$ROLLOUT_IP" ]; then
  ROLLOUT_IP="$(node_ip "$ROLLOUT_NODE")"
fi

ROLLOUT_SERVER_URLS="$(build_rollout_server_urls_literal "$ROLLOUT_IP" "${ROLLOUT_PORTS[@]}")"
ROLLOUT_METRICS_ENDPOINT_SPECS="$(build_rollout_monitor_specs "$ROLLOUT_IP" "${ROLLOUT_PORTS[@]}")"

echo "Run topology"
echo "  job_id:               $JOB_ID"
echo "  head_node:            $HEAD_NODE ($HEAD_IP)"
echo "  rollout_node:         $ROLLOUT_NODE ($ROLLOUT_IP)"
echo "  trainer_nodes:        $TRAINER_NODES_CSV"
echo "  run_name:             $RUN_NAME"
echo "  log_dir:              $LOG_DIR"
echo "  run_artifact_dir:     $RUN_ARTIFACT_DIR"
echo "  docker_data_root:     $DOCKER_DATA_ROOT"
echo "  train_runtime_scratch:$TRAIN_RUNTIME_SCRATCH_ROOT"
echo "  rollout_runtime_scratch:$ROLLOUT_RUNTIME_SCRATCH_ROOT"
echo "  agent_timeout_sec:    $AGENT_TIMEOUT_SEC"
echo "  max_train_tasks:      $MAX_TRAIN_TASKS"
echo "  head_nofile_soft:     $HEAD_NOFILE_SOFT"
echo "  trainer_nofile_soft:  $TRAINER_NOFILE_SOFT"
echo "  rollout_nofile_soft:  $ROLLOUT_NOFILE_SOFT"
echo "  docker_nofile_soft:   $DOCKER_NOFILE_SOFT"
echo "  docker_inotify_instances:$DOCKER_INOTIFY_MAX_USER_INSTANCES"
echo "  harbor_agent_max_turns:$HARBOR_AGENT_MAX_TURNS"
echo "  thunderagent_watchdog_enabled:$THUNDERAGENT_WATCHDOG_ENABLED"

srun --jobid "$JOB_ID" --overlap --overcommit --immediate=10 -w "$HEAD_NODE" --ntasks=1 --nodes=1 --cpus-per-task=4 --gres=gpu:0 \
  bash -lc "export PATH='$REPO_ROOT/.venv/bin':\$HOME/.local/bin:\$PATH && export XDG_RUNTIME_DIR='$XDG_RUNTIME_DIR' && export DOCKER_HOST='$DOCKER_HOST' && export DOCKER_PIDFILE='$DOCKER_PIDFILE' && export DOCKER_EXEC_ROOT='$DOCKER_EXEC_ROOT' && export DOCKER_DATA_ROOT='$DOCKER_DATA_ROOT' && export DOCKER_LOG_PATH='$DOCKER_LOG_PATH' && export DOCKER_NOFILE_SOFT='$DOCKER_NOFILE_SOFT' && export DOCKER_INOTIFY_MAX_USER_INSTANCES='$DOCKER_INOTIFY_MAX_USER_INSTANCES' && export SCRATCH_ROOT='$ROOTLESS_SCRATCH_ROOT' && mkdir -p '$XDG_RUNTIME_DIR' '$DOCKER_EXEC_ROOT' '$DOCKER_DATA_ROOT' '$ROOTLESS_SCRATCH_ROOT' && chmod 700 '$XDG_RUNTIME_DIR' && cd '$REPO_ROOT' && export ROOTLESS_DOCKER_START_MODE=block && bash '$SCRIPT_DIR/start_rootless_docker_for_harbor.sh'" \
  >"$ROOTLESS_DOCKER_LOG" 2>&1 &
PIDS+=("$!")

wait_for_head_docker "$HEAD_DOCKER_READY_TIMEOUT_SEC"
prepare_shared_mini_swe_agent_tool_home

JOB_ID="$JOB_ID" \
HEAD_NODE="$HEAD_NODE" \
TRAINER_NODES_CSV="$TRAINER_NODES_CSV" \
RAY_PORT="$RAY_PORT" \
RAY_START_MODE=block \
HEAD_NOFILE_SOFT="$HEAD_NOFILE_SOFT" \
TRAINER_NOFILE_SOFT="$TRAINER_NOFILE_SOFT" \
bash "$SCRIPT_DIR/launch_qwen3_32b_ray_cluster.sh" \
  >"$RAY_LOG" 2>&1 &
PIDS+=("$!")

wait_for_ray 180

srun --jobid "$JOB_ID" --overlap --overcommit --immediate=10 -w "$ROLLOUT_NODE" --ntasks=1 --nodes=1 --cpus-per-task=100 --gres=gpu:8 \
  bash -lc "cd '$REPO_ROOT' && export PATH='$REPO_ROOT/.venv/bin':\$HOME/.local/bin:\$PATH && export RUN_NAME='$RUN_NAME' && export RAY_HEAD_IP='$HEAD_IP' && export ROLLOUT_SERVER_PORTS_CSV='$ROLLOUT_SERVER_PORTS_CSV' && export TP_SIZE='$ROLLOUT_TP_SIZE' && export ROLLOUT_NOFILE_SOFT='$ROLLOUT_NOFILE_SOFT' && export LOG_DIR='$ROLLOUT_LOG_DIR' && export MONITORING_DIR='$ROLLOUT_MONITOR_DIR' && export TENSORBOARD_DIR='$ROLLOUT_LOG_DIR/tensorboard' && export SCRATCH_ROOT='$ROLLOUT_RUNTIME_SCRATCH_ROOT' && bash '$SCRIPT_DIR/start_qwen3_32b_external_rollout_servers.sh'" \
  >"$ROLLOUT_LOG" 2>&1 &
PIDS+=("$!")

suffixes=(a b c d e f g h i j k l m n o p)
for idx in "${!ROLLOUT_PORTS[@]}"; do
  wait_for_rollout_health "$ROLLOUT_IP" "${ROLLOUT_PORTS[$idx]}" "rollout_${suffixes[$idx]}" 900
done

JOB_ID="$JOB_ID" \
TRAINER_NODES_CSV="$TRAINER_NODES_CSV" \
RUN_NAME="$RUN_NAME" \
MONITOR_ROOT="$TRAINER_MONITOR_ROOT" \
TRAIN_LOG_DIR="$LOG_DIR" \
MONITOR_INTERVAL_SEC="$MONITOR_INTERVAL_SEC" \
bash "$SCRIPT_DIR/start_trainer_node_monitors.sh" \
  >"$TRAINER_MONITOR_LOG" 2>&1 &
PIDS+=("$!")

if [[ "$THUNDERAGENT_WATCHDOG_ENABLED" == "1" || "$THUNDERAGENT_WATCHDOG_ENABLED" == "true" ]]; then
  start_thunderagent_watchdog
fi

set +e
RAY_ADDRESS="$HEAD_IP:$RAY_PORT" \
RAY_HEAD_IP="$HEAD_IP" \
ROLLOUT_HOST_IP="$ROLLOUT_IP" \
TRAIN_DATA="$TRAIN_DATA" \
EVAL_DATA="$EVAL_DATA" \
ROLLOUT_SERVER_PORTS_CSV="$ROLLOUT_SERVER_PORTS_CSV" \
ROLLOUT_ENGINES="$ROLLOUT_ENGINES" \
ROLLOUT_TP_SIZE="$ROLLOUT_TP_SIZE" \
DOCKER_MODE="$DOCKER_MODE" \
RUN_NAME_OVERRIDE="$RUN_NAME" \
LOG_DIR_OVERRIDE="$LOG_DIR" \
RUN_ARTIFACT_ROOT="$RUN_ARTIFACT_ROOT" \
SCRATCH_ROOT="$TRAIN_RUNTIME_SCRATCH_ROOT" \
XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR" \
DOCKER_HOST="$DOCKER_HOST" \
DOCKER_PIDFILE="$DOCKER_PIDFILE" \
DOCKER_EXEC_ROOT="$DOCKER_EXEC_ROOT" \
DOCKER_DATA_ROOT="$DOCKER_DATA_ROOT" \
HEAD_NOFILE_SOFT="$HEAD_NOFILE_SOFT" \
ROLLOUT_METRICS_ENDPOINT_SPECS="$ROLLOUT_METRICS_ENDPOINT_SPECS" \
HARBOR_SHARED_UV_CACHE_HOST_DIR="$HARBOR_SHARED_UV_CACHE_HOST_DIR" \
HARBOR_SHARED_UV_CACHE_ENV_DIR="$HARBOR_SHARED_UV_CACHE_ENV_DIR" \
HARBOR_SHARED_MINI_SWE_TOOL_HOST_HOME="$HARBOR_SHARED_MINI_SWE_TOOL_HOST_HOME" \
HARBOR_SHARED_MINI_SWE_TOOL_ENV_HOME="$HARBOR_SHARED_MINI_SWE_TOOL_ENV_HOME" \
HARBOR_SHARED_UV_PYTHON_HOST_DIR="$HARBOR_SHARED_UV_PYTHON_HOST_DIR" \
HARBOR_SHARED_UV_PYTHON_ENV_DIR="$HARBOR_SHARED_UV_PYTHON_ENV_DIR" \
HARBOR_MINI_SWE_AGENT_GIT_REF="$HARBOR_MINI_SWE_AGENT_GIT_REF" \
HARBOR_MINI_SWE_AGENT_UV_OFFLINE="$HARBOR_MINI_SWE_AGENT_UV_OFFLINE" \
srun --jobid "$JOB_ID" --overlap --overcommit --immediate=10 -w "$HEAD_NODE" --ntasks=1 --nodes=1 --cpus-per-task=8 --gres=gpu:0 \
  bash -lc "cd '$REPO_ROOT' && export PATH='$REPO_ROOT/.venv/bin':\$HOME/.local/bin:\$PATH && stdbuf -oL -eL bash '$SCRIPT_DIR/run_codecontest_qwen3_32b_6node_rootless_fully_async.sh' full max_train_tasks='$MAX_TRAIN_TASKS' trainer.resume_mode=none harbor_trial_config.agent.name=mini-swe-agent harbor_trial_config.agent.kwargs.max_turns='$HARBOR_AGENT_MAX_TURNS' harbor_trial_config.agent.kwargs.llm_kwargs.timeout='$MINI_SWE_MODEL_TIMEOUT_SEC' harbor_trial_config.agent.override_timeout_sec='$AGENT_TIMEOUT_SEC'" \
  >"$TRAIN_DRIVER_LOG" 2>&1
TRAIN_RC=$?
set -e

sleep 5
cleanup_background
cleanup_new_steps

run_analysis

echo "Run finished with train_rc=$TRAIN_RC"
echo "  train_driver_log: $TRAIN_DRIVER_LOG"
echo "  summary_log:      $SUMMARY_LOG"
echo "  analysis_log:     $ANALYSIS_LOG"
echo "  analysis_dir:     $ANALYSIS_DIR"
echo "  run_artifact_dir: $RUN_ARTIFACT_DIR"

exit "$TRAIN_RC"
