#!/usr/bin/env bash
set -euo pipefail

COMPOSE_VERSION="${COMPOSE_VERSION:-v5.1.0}"
INSTALL_ROOT="${INSTALL_ROOT:-$HOME/.docker/cli-plugins}"
TARGET_BIN="$INSTALL_ROOT/docker-compose"

require_cmd() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || {
    echo "Missing required command: $cmd" >&2
    exit 1
  }
}

require_cmd curl
require_cmd docker

mkdir -p "$INSTALL_ROOT"
curl -fL "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-linux-x86_64" -o "$TARGET_BIN"
chmod +x "$TARGET_BIN"

docker compose version
