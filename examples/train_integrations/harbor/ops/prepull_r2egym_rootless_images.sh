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
    mkdir -p '$LOG_DIR' '$XDG_RUNTIME_DIR' '$DOCKER_EXEC_ROOT' '$DOCKER_DATA_ROOT' '$SCRATCH_ROOT' '$DOCKER_CONFIG'
    chmod 700 '$XDG_RUNTIME_DIR'
    cd '$REPO_ROOT'
    started_daemon=0
    if ! timeout 10 docker info >/dev/null 2>&1; then
      nohup \"\$HOME/.local/bin/dockerd-rootless.sh\" \
        --data-root '$DOCKER_DATA_ROOT' \
        --exec-root '$DOCKER_EXEC_ROOT' \
        --pidfile '$DOCKER_PIDFILE' \
        --host '$DOCKER_HOST' \
        --exec-opt native.cgroupdriver=cgroupfs \
        --default-address-pool base=10.240.0.0/12,size=24 \
        >'$DOCKER_LOG_PATH' 2>&1 &
      started_daemon=1
      deadline=\$((SECONDS + 30))
      until [ -S '${DOCKER_HOST#unix://}' ]; do
        if [ \$SECONDS -ge \$deadline ]; then
          echo 'Timed out waiting for rootless Docker socket' >&2
          exit 1
        fi
        sleep 1
      done
      docker info >/dev/null
    fi
    printf '%s' '$DOCKER_TOKEN' | docker login --username '$DOCKER_USERNAME' --password-stdin >/dev/null
    cleanup() {
      docker logout >/dev/null 2>&1 || true
      if [ \"\$started_daemon\" = 1 ] && [ -f '$DOCKER_PIDFILE' ]; then
        kill \"\$(cat '$DOCKER_PIDFILE')\" >/dev/null 2>&1 || true
      fi
    }
    trap cleanup EXIT
    cat '$IMAGE_LIST_PATH' | xargs -r -n 1 -P '$PULL_CONCURRENCY' docker pull
    docker image ls --format '{{.Repository}}:{{.Tag}}' | wc -l
  " | tee "$LOG_DIR/prepull.log"

echo "Pre-pull finished."
echo "  image_list: $IMAGE_LIST_PATH"
echo "  log_dir:    $LOG_DIR"
echo "  data_root:  $DOCKER_DATA_ROOT"
