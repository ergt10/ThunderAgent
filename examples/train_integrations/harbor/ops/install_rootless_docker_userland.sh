#!/usr/bin/env bash
set -euo pipefail

INSTALL_BIN_DIR="${INSTALL_BIN_DIR:-$HOME/.local/bin}"
INSTALL_TMP_DIR="${INSTALL_TMP_DIR:-$HOME/.cache/rootless-docker-userland}"
ROOTLESSKIT_DEB_URL="${ROOTLESSKIT_DEB_URL:-}"
MOBY_TAG="${MOBY_TAG:-v28.2.2}"

mkdir -p "$INSTALL_BIN_DIR" "$INSTALL_TMP_DIR"

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

download_rootlesskit_deb() {
  local deb_path="$INSTALL_TMP_DIR/rootlesskit.deb"

  if [ -n "$ROOTLESSKIT_DEB_URL" ]; then
    curl -fsSL "$ROOTLESSKIT_DEB_URL" -o "$deb_path"
    printf '%s\n' "$deb_path"
    return
  fi

  if have_cmd apt; then
    (
      cd "$INSTALL_TMP_DIR"
      rm -f rootlesskit_*.deb
      apt download rootlesskit >/dev/null
    )
    deb_path="$(find "$INSTALL_TMP_DIR" -maxdepth 1 -name 'rootlesskit_*.deb' | sort | tail -n 1)"
    if [ -n "$deb_path" ]; then
      printf '%s\n' "$deb_path"
      return
    fi
  fi

  echo "Failed to obtain rootlesskit .deb" >&2
  exit 1
}

install_rootlesskit() {
  local deb_path="$1"
  local extract_dir="$INSTALL_TMP_DIR/rootlesskit-extract"

  rm -rf "$extract_dir"
  mkdir -p "$extract_dir"
  dpkg-deb -x "$deb_path" "$extract_dir"

  if [ ! -x "$extract_dir/usr/bin/rootlesskit" ]; then
    echo "rootlesskit binary not found in $deb_path" >&2
    exit 1
  fi

  install -m 0755 "$extract_dir/usr/bin/rootlesskit" "$INSTALL_BIN_DIR/rootlesskit"
}

install_moby_rootless_scripts() {
  local base_url="https://raw.githubusercontent.com/moby/moby/${MOBY_TAG}/contrib"
  curl -fsSL "$base_url/dockerd-rootless.sh" -o "$INSTALL_BIN_DIR/dockerd-rootless.sh"
  curl -fsSL "$base_url/dockerd-rootless-setuptool.sh" -o "$INSTALL_BIN_DIR/dockerd-rootless-setuptool.sh"
  chmod 0755 "$INSTALL_BIN_DIR/dockerd-rootless.sh" "$INSTALL_BIN_DIR/dockerd-rootless-setuptool.sh"
}

deb_path="$(download_rootlesskit_deb)"
install_rootlesskit "$deb_path"
install_moby_rootless_scripts

echo "Installed user-space rootless Docker helpers:"
echo "  $INSTALL_BIN_DIR/rootlesskit"
echo "  $INSTALL_BIN_DIR/dockerd-rootless.sh"
echo "  $INSTALL_BIN_DIR/dockerd-rootless-setuptool.sh"
echo "Ensure PATH includes: $INSTALL_BIN_DIR"
