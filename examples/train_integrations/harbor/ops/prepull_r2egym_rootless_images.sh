#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"

JOB_ID="${JOB_ID:-${SLURM_JOB_ID:-}}"
HEAD_NODE="${HEAD_NODE:-research-dev-coder-003}"
PULL_CONCURRENCY="${PULL_CONCURRENCY:-8}"
RUN_NAME="${RUN_NAME:-r2egym-rootless-image-prepull-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$WORKSPACE_ROOT/tmp_logs/$RUN_NAME}"

DATASET_ROOTS=(
  "${R2EGYM_TRIVIAL_ROOT:-/home/hkang/zthunder_agent/data/harbor/r2egym-trivial}"
  "${R2EGYM_EASY_ROOT:-/home/hkang/zthunder_agent/data/harbor/r2egym-easy}"
  "${R2EGYM_MEDIUM_ROOT:-/home/hkang/zthunder_agent/data/harbor/r2egym-medium}"
  "${R2EGYM_HARD_ROOT:-/home/hkang/zthunder_agent/data/harbor/r2egym-hard}"
)

DEFAULT_XDG_RUNTIME_DIR="/tmp/xdg-r2egym-image-cache-$USER"
case "${XDG_RUNTIME_DIR:-}" in
  ""|/run/user/*)
    XDG_RUNTIME_DIR="$DEFAULT_XDG_RUNTIME_DIR"
    ;;
esac
DOCKER_HOST="${DOCKER_HOST:-unix://$XDG_RUNTIME_DIR/docker.sock}"
DOCKER_PIDFILE="${DOCKER_PIDFILE:-$XDG_RUNTIME_DIR/docker.pid}"
DOCKER_EXEC_ROOT="${DOCKER_EXEC_ROOT:-/tmp/$USER/r2egym-rootless-image-cache-exec}"
DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT:-/scratch/triton_cache/$USER/r2egym-rootless-image-cache}"
DOCKER_LOG_PATH="${DOCKER_LOG_PATH:-$LOG_DIR/dockerd_rootless_harbor.log}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/tmp/$USER/r2egym-rootless-image-cache-scratch}"
DOCKER_NOFILE_SOFT="${DOCKER_NOFILE_SOFT:-131072}"
DOCKER_CONFIG="${DOCKER_CONFIG:-$LOG_DIR/docker-config}"
PREPULL_RESTART_VERIFY_VISIBILITY="${PREPULL_RESTART_VERIFY_VISIBILITY:-true}"
DOCKER_SHUTDOWN_TIMEOUT_SEC="${DOCKER_SHUTDOWN_TIMEOUT_SEC:-30}"
PREPULL_VERIFY_ROOT_BASE="${PREPULL_VERIFY_ROOT_BASE:-/tmp/$USER/r2e-rv-$(date +%m%d%H%M%S)}"
IMAGE_LIST_PATH="$LOG_DIR/r2egym_images.txt"

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

require_cmd srun
require_env JOB_ID
require_env DOCKER_USERNAME
require_env DOCKER_TOKEN

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python env not found: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

"$PYTHON_BIN" - <<'PY' "${DATASET_ROOTS[@]}" >"$IMAGE_LIST_PATH"
from pathlib import Path
import sys
import tomllib

images = set()
for root_arg in sys.argv[1:]:
    root = Path(root_arg).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Dataset root not found: {root}")
    for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        task_toml = task_dir / "task.toml"
        config = tomllib.loads(task_toml.read_text(encoding="utf-8"))
        image = ((config.get("environment") or {}).get("docker_image"))
        if image:
            images.add(image)

for image in sorted(images):
    print(image)
PY

IMAGE_COUNT="$(wc -l <"$IMAGE_LIST_PATH" | tr -d ' ')"
echo "Prepared image list: $IMAGE_LIST_PATH"
echo "Unique images: $IMAGE_COUNT"
echo "Head node: $HEAD_NODE"
echo "Target DOCKER_DATA_ROOT: $DOCKER_DATA_ROOT"

srun --jobid "$JOB_ID" --overlap --overcommit --immediate=10 \
  -w "$HEAD_NODE" --ntasks=1 --nodes=1 --cpus-per-task=8 --gres=gpu:0 \
  bash -lc "
    set -euo pipefail
    export PATH='$REPO_ROOT/.venv/bin':\$HOME/.local/bin:\$PATH
    export XDG_RUNTIME_DIR='$XDG_RUNTIME_DIR'
    export DOCKER_HOST='$DOCKER_HOST'
    export DOCKER_PIDFILE='$DOCKER_PIDFILE'
    export DOCKER_EXEC_ROOT='$DOCKER_EXEC_ROOT'
    export DOCKER_DATA_ROOT='$DOCKER_DATA_ROOT'
    export DOCKER_LOG_PATH='$DOCKER_LOG_PATH'
    export SCRATCH_ROOT='$SCRATCH_ROOT'
    export DOCKER_NOFILE_SOFT='$DOCKER_NOFILE_SOFT'
    export DOCKER_CONFIG='$DOCKER_CONFIG'
    export PREPULL_RESTART_VERIFY_VISIBILITY='$PREPULL_RESTART_VERIFY_VISIBILITY'
    export DOCKER_SHUTDOWN_TIMEOUT_SEC='$DOCKER_SHUTDOWN_TIMEOUT_SEC'
    export PREPULL_VERIFY_ROOT_BASE='$PREPULL_VERIFY_ROOT_BASE'
    VERIFY_XDG_RUNTIME_DIR=\"\${VERIFY_XDG_RUNTIME_DIR:-\${PREPULL_VERIFY_ROOT_BASE}/xdg}\"
    VERIFY_DOCKER_HOST=\"\${VERIFY_DOCKER_HOST:-unix://\${VERIFY_XDG_RUNTIME_DIR}/docker.sock}\"
    VERIFY_DOCKER_PIDFILE=\"\${VERIFY_DOCKER_PIDFILE:-\${VERIFY_XDG_RUNTIME_DIR}/docker.pid}\"
    VERIFY_DOCKER_EXEC_ROOT=\"\${VERIFY_DOCKER_EXEC_ROOT:-\${PREPULL_VERIFY_ROOT_BASE}/exec}\"
    VERIFY_SCRATCH_ROOT=\"\${VERIFY_SCRATCH_ROOT:-\${PREPULL_VERIFY_ROOT_BASE}/scratch}\"
    VERIFY_DOCKER_LOG_PATH=\"\${VERIFY_DOCKER_LOG_PATH:-${LOG_DIR}/dockerd_rootless_harbor.restart_verify.log}\"
    RESTART_VERIFY_MISSING_PATH='${LOG_DIR}/restart_visibility_missing.txt'
    RESTART_VERIFY_COUNT_PATH='${LOG_DIR}/restart_visibility_count.txt'
    mkdir -p '$LOG_DIR' '$XDG_RUNTIME_DIR' '$DOCKER_EXEC_ROOT' '$DOCKER_DATA_ROOT' '$SCRATCH_ROOT' '$DOCKER_CONFIG' \"\$VERIFY_XDG_RUNTIME_DIR\" \"\$VERIFY_DOCKER_EXEC_ROOT\" \"\$VERIFY_SCRATCH_ROOT\"
    chmod 700 '$XDG_RUNTIME_DIR' \"\$VERIFY_XDG_RUNTIME_DIR\"
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
      mkdir -p \"\$xdg_runtime_dir\" \"\$docker_exec_root\" '$DOCKER_DATA_ROOT' \"\$scratch_root\"
      chmod 700 \"\$xdg_runtime_dir\"
      XDG_RUNTIME_DIR=\"\$xdg_runtime_dir\" \
      DOCKER_HOST=\"\$docker_host\" \
      DOCKER_PIDFILE=\"\$docker_pidfile\" \
      DOCKER_EXEC_ROOT=\"\$docker_exec_root\" \
      DOCKER_DATA_ROOT='$DOCKER_DATA_ROOT' \
      DOCKER_LOG_PATH=\"\$docker_log_path\" \
      SCRATCH_ROOT=\"\$scratch_root\" \
      DOCKER_NOFILE_SOFT='$DOCKER_NOFILE_SOFT' \
      bash '$REPO_ROOT/examples/train_integrations/harbor/start_rootless_docker_for_harbor.sh' >/dev/null
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
      start_rootless_daemon '$XDG_RUNTIME_DIR' '$DOCKER_HOST' '$DOCKER_PIDFILE' '$DOCKER_EXEC_ROOT' '$DOCKER_LOG_PATH' '$SCRATCH_ROOT'
      primary_started_by_script=1
    fi

    cleanup() {
      docker logout >/dev/null 2>&1 || true
      if [ \"\$verify_started_by_script\" = 1 ]; then
        stop_rootless_daemon \"\$VERIFY_DOCKER_PIDFILE\" \"\$VERIFY_DOCKER_HOST\" '$DOCKER_SHUTDOWN_TIMEOUT_SEC'
      fi
      if [ \"\$primary_started_by_script\" = 1 ]; then
        stop_rootless_daemon '$DOCKER_PIDFILE' '$DOCKER_HOST' '$DOCKER_SHUTDOWN_TIMEOUT_SEC'
      fi
    }
    trap cleanup EXIT
    printf '%s' '$DOCKER_TOKEN' | docker login --username '$DOCKER_USERNAME' --password-stdin >/dev/null
    cat '$IMAGE_LIST_PATH' | xargs -r -n 1 -P '$PULL_CONCURRENCY' docker pull
    if [ \"\$PREPULL_RESTART_VERIFY_VISIBILITY\" = true ]; then
      if [ \"\$primary_started_by_script\" != 1 ]; then
        echo 'Restart visibility verification requires the prepull script to own the Docker daemon lifecycle for this DOCKER_HOST.' >&2
        echo 'Refusing to stop a pre-existing daemon.' >&2
        exit 1
      fi
      stop_rootless_daemon '$DOCKER_PIDFILE' '$DOCKER_HOST' '$DOCKER_SHUTDOWN_TIMEOUT_SEC'
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
    else
      docker image ls --format '{{.Repository}}:{{.Tag}}' | wc -l
    fi
  " | tee "$LOG_DIR/prepull.log"

echo "Pre-pull finished."
echo "  image_list: $IMAGE_LIST_PATH"
echo "  log_dir:    $LOG_DIR"
echo "  data_root:  $DOCKER_DATA_ROOT"
