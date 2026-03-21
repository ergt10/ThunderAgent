#!/usr/bin/env bash
set -euo pipefail

require_cmd() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || {
    echo "Missing required command: $cmd" >&2
    exit 1
  }
}

is_usable_ifname() {
  local ifname="${1:-}"
  [ -n "$ifname" ] || return 1
  case "$ifname" in
    lo|docker*|br-*|veth*)
      return 1
      ;;
  esac
  return 0
}

detect_via_route() {
  local target_ip="$1"
  ip route get "$target_ip" 2>/dev/null | awk '
    {
      for (i = 1; i <= NF; i++) {
        if ($i == "dev" && (i + 1) <= NF) {
          print $(i + 1)
          exit
        }
      }
    }
  '
}

require_cmd ip

TARGET_IP="${1:-}"
SOCKET_IFNAME=""

if [ -n "$TARGET_IP" ]; then
  SOCKET_IFNAME="$(detect_via_route "$TARGET_IP" || true)"
fi

if ! is_usable_ifname "$SOCKET_IFNAME"; then
  SOCKET_IFNAME=""
fi

if [ -z "$SOCKET_IFNAME" ]; then
  PRIMARY_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
  if [ -n "$PRIMARY_IP" ]; then
    SOCKET_IFNAME="$(
      ip -o -4 addr show up scope global \
        | awk -v primary_ip="$PRIMARY_IP" '$4 ~ ("^" primary_ip "/") { print $2; exit }'
    )"
  fi
fi

if ! is_usable_ifname "$SOCKET_IFNAME"; then
  SOCKET_IFNAME=""
fi

if [ -z "$SOCKET_IFNAME" ]; then
  SOCKET_IFNAME="$(
    ip -o -4 addr show up scope global \
      | awk '$2 !~ /^(lo|docker|br-|veth)/ { print $2; exit }'
  )"
fi

if [ -z "$SOCKET_IFNAME" ]; then
  echo "Failed to detect a usable socket interface name" >&2
  exit 1
fi

printf '%s\n' "$SOCKET_IFNAME"
