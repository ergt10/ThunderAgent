#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"

JOB_ID="${JOB_ID:-}"
ROLLOUT_NODE="${ROLLOUT_NODE:-}"
SRUN_RETRIES="${SRUN_RETRIES:-5}"
SRUN_RETRY_DELAY_SEC="${SRUN_RETRY_DELAY_SEC:-2}"
ROLLOUT_CUDA_SMOKE_TIMEOUT_SEC="${ROLLOUT_CUDA_SMOKE_TIMEOUT_SEC:-240}"
ROLLOUT_CUDA_SMOKE_GPUS="${ROLLOUT_CUDA_SMOKE_GPUS:-4}"
ROLLOUT_CUDA_VISIBLE_DEVICES="${ROLLOUT_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
RUN_NAME="${RUN_NAME:-rollout-cuda-runtime-smoke-$(date +%s)}"
LOG_DIR="${LOG_DIR:-$WORKSPACE_ROOT/tmp_logs/$RUN_NAME}"
LOG_FILE="$LOG_DIR/rollout_cuda_runtime_smoke.log"

require_env() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "Missing required environment variable: $name" >&2
    exit 1
  fi
}

require_env JOB_ID
require_env ROLLOUT_NODE

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python env not found: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

run_smoke() {
  local attempt=1
  local rc=0
  local output_file=""
  while [ "$attempt" -le "$SRUN_RETRIES" ]; do
    output_file="$(mktemp)"
    if timeout "$ROLLOUT_CUDA_SMOKE_TIMEOUT_SEC" \
      srun --jobid "$JOB_ID" --overlap --overcommit --immediate=10 \
      -w "$ROLLOUT_NODE" --ntasks=1 --nodes=1 --cpus-per-task=8 \
      --gres="gpu:${ROLLOUT_CUDA_SMOKE_GPUS}" \
      bash -lc "cd '$REPO_ROOT' && export CUDA_VISIBLE_DEVICES='$ROLLOUT_CUDA_VISIBLE_DEVICES' && export PYTHONUNBUFFERED=1 && TMP_SCRIPT=\$(mktemp) && cat >\"\$TMP_SCRIPT\" <<'PY'
import json
import multiprocessing as mp
import os
import traceback

import torch


def report(obj):
    print(json.dumps(obj, sort_keys=True), flush=True)


def single_process_checks():
    status = {
        'phase': 'single_process',
        'cuda_is_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count(),
        'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        'torch_version': torch.__version__,
        'python_version': os.sys.version,
    }
    report(status)
    if not status['cuda_is_available']:
        raise SystemExit('single-process torch.cuda.is_available() returned false')
    for idx in range(torch.cuda.device_count()):
        report({
            'phase': 'single_process_device',
            'device': idx,
            'name': torch.cuda.get_device_name(idx),
            'capability': torch.cuda.get_device_capability(idx),
        })


def worker(rank, queue):
    try:
        import torch
        torch.cuda.set_device(rank)
        capability = torch.cuda.get_device_capability(rank)
        name = torch.cuda.get_device_name(rank)
        fa3_supported = None
        fa_probe_error = None
        try:
            from vllm.vllm_flash_attn.flash_attn_interface import is_fa_version_supported
            fa3_supported = is_fa_version_supported(3)
        except Exception as exc:  # pragma: no cover - diagnostic path
            fa_probe_error = ''.join(traceback.format_exception_only(type(exc), exc)).strip()
        queue.put({
            'phase': 'spawn_worker',
            'rank': rank,
            'ok': True,
            'device_name': name,
            'capability': capability,
            'fa3_supported': fa3_supported,
            'fa_probe_error': fa_probe_error,
        })
    except Exception as exc:  # pragma: no cover - diagnostic path
        queue.put({
            'phase': 'spawn_worker',
            'rank': rank,
            'ok': False,
            'error': ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        })


def spawn_checks(num_workers):
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(rank, queue)) for rank in range(num_workers)]
    for proc in procs:
        proc.start()
    results = [queue.get(timeout=180) for _ in procs]
    for proc in procs:
        proc.join(timeout=30)
    for result in sorted(results, key=lambda item: item['rank']):
        report(result)
    bad = [r for r in results if not r['ok']]
    if bad:
        raise SystemExit('spawn worker CUDA checks failed')


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    single_process_checks()
    spawn_checks(4)
    print('CUDA_RUNTIME_SMOKE_OK', flush=True)
PY
'$PYTHON_BIN' \"\$TMP_SCRIPT\" >'$LOG_FILE' 2>&1
rc=\$?
rm -f \"\$TMP_SCRIPT\"
exit \$rc" >"$output_file" 2>&1; then
      if grep -qx 'CUDA_RUNTIME_SMOKE_OK' "$LOG_FILE"; then
        cat "$output_file"
        rm -f "$output_file"
        return 0
      fi
      cat "$output_file" >&2
      echo "CUDA runtime smoke did not emit success sentinel" >&2
      rm -f "$output_file"
      return 1
    fi
    rc=$?
    cat "$output_file" >&2
    if ! grep -Eq "Requested nodes are busy|step creation temporarily disabled" "$output_file"; then
      rm -f "$output_file"
      return "$rc"
    fi
    rm -f "$output_file"
    if [ "$attempt" -ge "$SRUN_RETRIES" ]; then
      return "$rc"
    fi
    sleep "$SRUN_RETRY_DELAY_SEC"
    attempt=$((attempt + 1))
  done
  return "$rc"
}

echo "Rollout CUDA runtime smoke"
echo "  job_id:       $JOB_ID"
echo "  rollout_node: $ROLLOUT_NODE"
echo "  log_file:     $LOG_FILE"

smoke_rc=0
if run_smoke; then
  smoke_rc=0
else
  smoke_rc=$?
fi

echo
echo "Smoke log"
cat "$LOG_FILE"

if ! grep -qx 'CUDA_RUNTIME_SMOKE_OK' "$LOG_FILE"; then
  exit 1
fi

exit "$smoke_rc"
