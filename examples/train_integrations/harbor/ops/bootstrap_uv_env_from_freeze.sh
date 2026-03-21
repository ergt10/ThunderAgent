#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
UV_INSTALL_DIR="${UV_INSTALL_DIR:-$HOME/.local/bin}"
PYTHON_VERSION="${PYTHON_VERSION:-3.13}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
FREEZE_FILE="${FREEZE_FILE:-$REPO_ROOT/requirements.txt}"
SANITIZED_FILE="${SANITIZED_FILE:-$REPO_ROOT/.codex/generated/requirements.uv.txt}"
APPLY_HARBOR_ROOTLESS_PATCH="${APPLY_HARBOR_ROOTLESS_PATCH:-true}"

ensure_uv() {
  if [ -x "$UV_BIN" ]; then
    return 0
  fi

  mkdir -p "$UV_INSTALL_DIR"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$UV_INSTALL_DIR" sh
}

ensure_uv

"$UV_BIN" python install "$PYTHON_VERSION"
python3 "$SCRIPT_DIR/sanitize_freeze_for_uv.py" \
  --input "$FREEZE_FILE" \
  --output "$SANITIZED_FILE" \
  --repo-root "$REPO_ROOT"

"$UV_BIN" venv --clear --python "$PYTHON_VERSION" "$VENV_DIR"
"$UV_BIN" pip install \
  --python "$VENV_DIR/bin/python" \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://flashinfer.ai/whl/cu128 \
  --index-strategy unsafe-best-match \
  -r "$SANITIZED_FILE"

if [ "$APPLY_HARBOR_ROOTLESS_PATCH" = true ]; then
  "$VENV_DIR/bin/python" "$SCRIPT_DIR/apply_harbor_rootless_patch.py" --backup
fi

echo "uv environment ready:"
echo "  uv:   $UV_BIN"
echo "  venv: $VENV_DIR"
echo "  reqs: $SANITIZED_FILE"
