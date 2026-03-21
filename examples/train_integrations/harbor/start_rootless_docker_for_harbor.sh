#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export PATH="$HOME/.local/bin:$PATH"

XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/xdg-test-$USER}"
DOCKER_HOST="${DOCKER_HOST:-unix://$XDG_RUNTIME_DIR/docker.sock}"
DOCKER_PIDFILE="${DOCKER_PIDFILE:-$XDG_RUNTIME_DIR/docker.pid}"
DOCKER_EXEC_ROOT="${DOCKER_EXEC_ROOT:-$XDG_RUNTIME_DIR/docker-exec}"
STARTUP_TIMEOUT_SEC="${STARTUP_TIMEOUT_SEC:-30}"
ROOTLESS_DOCKER_START_MODE="${ROOTLESS_DOCKER_START_MODE:-background}"
DOCKER_DEFAULT_ADDRESS_POOL_BASE="${DOCKER_DEFAULT_ADDRESS_POOL_BASE:-10.240.0.0/12}"
DOCKER_DEFAULT_ADDRESS_POOL_SIZE="${DOCKER_DEFAULT_ADDRESS_POOL_SIZE:-24}"
DOCKER_NOFILE_SOFT="${DOCKER_NOFILE_SOFT:-131072}"
DOCKER_INOTIFY_MAX_USER_INSTANCES="${DOCKER_INOTIFY_MAX_USER_INSTANCES:-}"

require_cmd() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || {
    echo "Missing required command: $cmd" >&2
    exit 1
  }
}

ensure_soft_nofile_limit() {
  local requested_soft="$1"
  local current_soft=""
  local current_hard=""

  if [ -z "$requested_soft" ]; then
    return 0
  fi
  if ! [[ "$requested_soft" =~ ^[0-9]+$ ]]; then
    echo "Invalid DOCKER_NOFILE_SOFT: $requested_soft" >&2
    exit 1
  fi

  current_soft="$(ulimit -Sn)"
  current_hard="$(ulimit -Hn)"
  if [ "$current_soft" != "unlimited" ] && [ "$current_soft" -lt "$requested_soft" ]; then
    if ! ulimit -Sn "$requested_soft" 2>/dev/null; then
      echo "Failed to raise soft nofile from $current_soft to $requested_soft (hard=$current_hard)." >&2
      echo "Raise the caller's RLIMIT_NOFILE before starting rootless Docker." >&2
      exit 1
    fi
  fi

  current_soft="$(ulimit -Sn)"
  if [ "$current_soft" != "unlimited" ] && [ "$current_soft" -lt "$requested_soft" ]; then
    echo "Soft nofile is still below the requested threshold after attempting to raise it: $current_soft < $requested_soft" >&2
    exit 1
  fi
}

ensure_inotify_instances_limit() {
  local requested="$1"
  local current=""

  if [ -z "$requested" ]; then
    return 0
  fi
  if ! [[ "$requested" =~ ^[0-9]+$ ]]; then
    echo "Invalid DOCKER_INOTIFY_MAX_USER_INSTANCES: $requested" >&2
    exit 1
  fi

  current="$(sysctl -n fs.inotify.max_user_instances)"
  if [ "$current" -ge "$requested" ]; then
    return 0
  fi

  if ! sudo -n sysctl -w "fs.inotify.max_user_instances=$requested" >/dev/null; then
    echo "Failed to raise fs.inotify.max_user_instances from $current to $requested." >&2
    exit 1
  fi

  current="$(sysctl -n fs.inotify.max_user_instances)"
  if [ "$current" -lt "$requested" ]; then
    echo "fs.inotify.max_user_instances is still below the requested threshold after sysctl: $current < $requested" >&2
    exit 1
  fi
}

resolve_dockerd_rootless_bin() {
  if command -v dockerd-rootless.sh >/dev/null 2>&1; then
    command -v dockerd-rootless.sh
    return
  fi

  local candidate
  for candidate in \
    "$HOME/.local/bin/dockerd-rootless.sh" \
    /usr/bin/dockerd-rootless.sh \
    /usr/local/bin/dockerd-rootless.sh \
    /usr/libexec/docker/dockerd-rootless.sh \
    /usr/lib/docker/dockerd-rootless.sh; do
    if [ -x "$candidate" ]; then
      printf '%s\n' "$candidate"
      return
    fi
  done
}

resolve_writable_runtime_root() {
  local preferred="${1:-}"
  local candidate=""
  for candidate in \
    "$preferred" \
    "/tmp/$USER/docker-rootless" \
    "$HOME/.cache/docker-rootless"; do
    [ -n "$candidate" ] || continue
    if mkdir -p "$candidate" >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  echo "Failed to find a writable runtime root" >&2
  exit 1
}

require_cmd docker
require_cmd slirp4netns
require_cmd rootlesskit
require_cmd newuidmap
require_cmd newgidmap

SCRATCH_ROOT="$(resolve_writable_runtime_root "${SCRATCH_ROOT:-/scratch/$USER}")"
SCRATCH="${SCRATCH:-$SCRATCH_ROOT/scratch}"
DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT:-$SCRATCH_ROOT/docker-rootless}"
DOCKER_LOG_PATH="${DOCKER_LOG_PATH:-$SCRATCH/dockerd_rootless_harbor.log}"

DOCKERD_ROOTLESS_BIN="$(resolve_dockerd_rootless_bin)"
if [ -z "$DOCKERD_ROOTLESS_BIN" ] || [ ! -x "$DOCKERD_ROOTLESS_BIN" ]; then
  echo "Missing required command: dockerd-rootless.sh" >&2
  echo "The rootless Docker startup flow from the Harbor run notes requires Docker's rootless extras." >&2
  exit 1
fi

docker compose version >/dev/null 2>&1 || {
  echo "Missing required feature: docker compose" >&2
  echo "Harbor's Docker backend expects the docker compose CLI plugin to be available." >&2
  exit 1
}

mkdir -p "$SCRATCH" "$DOCKER_DATA_ROOT" "$XDG_RUNTIME_DIR"
rm -f "${DOCKER_HOST#unix://}"

export XDG_RUNTIME_DIR
export DOCKER_HOST
export DOCKER_DATA_ROOT
export DOCKER_PIDFILE
export DOCKER_EXEC_ROOT

ensure_soft_nofile_limit "$DOCKER_NOFILE_SOFT"
ensure_inotify_instances_limit "$DOCKER_INOTIFY_MAX_USER_INSTANCES"

DOCKER_DEFAULT_ADDRESS_POOL_ARGS=()
if [ -n "$DOCKER_DEFAULT_ADDRESS_POOL_BASE" ] && [ -n "$DOCKER_DEFAULT_ADDRESS_POOL_SIZE" ]; then
  DOCKER_DEFAULT_ADDRESS_POOL_ARGS=(
    --default-address-pool
    "base=${DOCKER_DEFAULT_ADDRESS_POOL_BASE},size=${DOCKER_DEFAULT_ADDRESS_POOL_SIZE}"
  )
fi

run_dockerd_rootless() {
  exec "$DOCKERD_ROOTLESS_BIN" \
    --data-root "$DOCKER_DATA_ROOT" \
    --exec-root "$DOCKER_EXEC_ROOT" \
    --pidfile "$DOCKER_PIDFILE" \
    --host "$DOCKER_HOST" \
    --exec-opt native.cgroupdriver=cgroupfs \
    "${DOCKER_DEFAULT_ADDRESS_POOL_ARGS[@]}"
}

if [ "$ROOTLESS_DOCKER_START_MODE" = "block" ]; then
  run_dockerd_rootless
fi

if [ -f "$DOCKER_PIDFILE" ] && kill -0 "$(cat "$DOCKER_PIDFILE")" 2>/dev/null; then
  echo "Rootless dockerd already running with pid $(cat "$DOCKER_PIDFILE")"
else
  nohup "$DOCKERD_ROOTLESS_BIN" \
    --data-root "$DOCKER_DATA_ROOT" \
    --exec-root "$DOCKER_EXEC_ROOT" \
    --pidfile "$DOCKER_PIDFILE" \
    --host "$DOCKER_HOST" \
    --exec-opt native.cgroupdriver=cgroupfs \
    "${DOCKER_DEFAULT_ADDRESS_POOL_ARGS[@]}" \
    >"$DOCKER_LOG_PATH" 2>&1 &
fi

deadline=$((SECONDS + STARTUP_TIMEOUT_SEC))
until [ -S "${DOCKER_HOST#unix://}" ]; do
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "Timed out waiting for rootless Docker socket at ${DOCKER_HOST#unix://}" >&2
    echo "Log: $DOCKER_LOG_PATH" >&2
    exit 1
  fi
  sleep 1
done

docker info >/dev/null

echo "Rootless Docker ready"
echo "  docker_host: $DOCKER_HOST"
echo "  data_root:   $DOCKER_DATA_ROOT"
echo "  pidfile:     $DOCKER_PIDFILE"
echo "  log:         $DOCKER_LOG_PATH"
echo "  nofile:      soft=$(ulimit -Sn) hard=$(ulimit -Hn)"
if [ "${#DOCKER_DEFAULT_ADDRESS_POOL_ARGS[@]}" -gt 0 ]; then
  echo "  addr_pool:   base=${DOCKER_DEFAULT_ADDRESS_POOL_BASE},size=${DOCKER_DEFAULT_ADDRESS_POOL_SIZE}"
fi
