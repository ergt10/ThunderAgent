#!/usr/bin/env bash
set -euo pipefail

# Minimal reproducer for the Harbor verifier failure under rootless Docker.
# It avoids Harbor entirely and only exercises:
#   docker compose cp <host_dir>/. main:/tests
#
# Expected failure on the rootless Harbor head node:
#   failed to Lchown "/tests" for UID <large_uid>, GID <large_gid> ...

ROOT_DIR="${ROOT_DIR:-/scratch/$USER/rootless-docker-cp-repro}"
PROJECT_NAME="${PROJECT_NAME:-rootless-cp-repro}"
COMPOSE_DIR="$ROOT_DIR/compose"
SRC_DIR="$ROOT_DIR/src-tests"
LOG_DIR="$ROOT_DIR/logs"
XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/xdg-test-$USER}"
DOCKER_HOST="${DOCKER_HOST:-unix://$XDG_RUNTIME_DIR/docker.sock}"

mkdir -p "$COMPOSE_DIR" "$SRC_DIR" "$LOG_DIR" "$XDG_RUNTIME_DIR"

cat >"$COMPOSE_DIR/compose.yaml" <<'YAML'
services:
  main:
    image: alpine:3.20
    command: ["sh", "-lc", "sleep infinity"]
YAML

cat >"$SRC_DIR/test.sh" <<'SH'
#!/usr/bin/env bash
echo ok
SH
chmod +x "$SRC_DIR/test.sh"

cat >"$SRC_DIR/test_data.json" <<'JSON'
{"ok": true}
JSON

cleanup() {
  docker compose -p "$PROJECT_NAME" --project-directory "$COMPOSE_DIR" down -v --remove-orphans >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "== identity =="
id
echo

echo "== source ownership =="
stat -c '%u %g %n' "$SRC_DIR" "$SRC_DIR"/*
echo

echo "== docker info =="
docker info --format '{{.ServerVersion}} {{.Driver}} {{.CgroupDriver}}'
echo

echo "== compose up =="
docker compose -p "$PROJECT_NAME" --project-directory "$COMPOSE_DIR" up -d
echo

echo "== reproducer =="
set +e
docker compose -p "$PROJECT_NAME" --project-directory "$COMPOSE_DIR" cp "$SRC_DIR/." main:/tests \
  >"$LOG_DIR/compose-cp.stdout.log" 2>"$LOG_DIR/compose-cp.stderr.log"
rc=$?
set -e

echo "compose cp rc=$rc"
echo

echo "== stdout =="
cat "$LOG_DIR/compose-cp.stdout.log" || true
echo

echo "== stderr =="
cat "$LOG_DIR/compose-cp.stderr.log" || true
echo

if [[ $rc -eq 0 ]]; then
  echo "UNEXPECTED: compose cp succeeded"
  exit 2
fi

if grep -Eq 'Lchown|lchown .*invalid argument|subordinate IDs' "$LOG_DIR/compose-cp.stdout.log" "$LOG_DIR/compose-cp.stderr.log"; then
  echo "EXPECTED FAILURE REPRODUCED"
  exit 0
fi

echo "compose cp failed, but not with the expected rootless chown signature"
exit 1
