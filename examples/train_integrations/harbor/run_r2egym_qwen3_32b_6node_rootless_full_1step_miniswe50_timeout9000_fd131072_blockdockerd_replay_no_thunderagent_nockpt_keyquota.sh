#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$REPO_ROOT"
export PATH="$REPO_ROOT/.venv/bin:$HOME/.local/bin:$PATH"

export JOB_ID="${JOB_ID:-${SLURM_JOB_ID:-}}"
export HEAD_NODE="${HEAD_NODE:-research-dev-coder-003}"
export KERNEL_KEYS_MAXKEYS="${KERNEL_KEYS_MAXKEYS:-20000}"
export KERNEL_KEYS_MAXBYTES="${KERNEL_KEYS_MAXBYTES:-25000000}"

if [[ -z "$JOB_ID" ]]; then
  echo "JOB_ID is required" >&2
  exit 1
fi

echo "[keyquota] raising kernel key quota on $HEAD_NODE for job $JOB_ID"
srun --jobid "$JOB_ID" --overlap --overcommit --immediate=10 -w "$HEAD_NODE" --ntasks=1 --nodes=1 \
  bash -lc "sudo sysctl -w kernel.keys.maxkeys=$KERNEL_KEYS_MAXKEYS kernel.keys.maxbytes=$KERNEL_KEYS_MAXBYTES && sysctl kernel.keys.maxkeys kernel.keys.maxbytes"

exec bash "$SCRIPT_DIR/run_r2egym_qwen3_32b_6node_rootless_full_1step_miniswe50_timeout9000_fd131072_blockdockerd_replay_no_thunderagent_nockpt.sh"
