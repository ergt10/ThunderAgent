#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:?RUN_NAME is required}"
JOB_ID="${JOB_ID:?JOB_ID is required}"
TARGET_STEP="${TARGET_STEP:-5}"
LOG_ROOT="${LOG_ROOT:-/home/hkang/zthunder_agent/tmp_logs}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/home/hkang/zthunder_agent/harbor_runs}"
REPO_ROOT="${REPO_ROOT:-/home/hkang/zthunder_agent/SkyRL}"
USER_NAME="${USER_NAME:-$USER}"

RUN_LOG_DIR="${LOG_ROOT}/${RUN_NAME}"
RUN_ARTIFACT_DIR="${ARTIFACT_ROOT}/${RUN_NAME}"
SUMMARY_DIR="${RUN_LOG_DIR}/postmortem_step${TARGET_STEP}"
SUMMARY_MD="${SUMMARY_DIR}/RUN_STOP_AT_STEP_${TARGET_STEP}_SUMMARY.md"
STATE_JSON="${SUMMARY_DIR}/watch_state.json"

mkdir -p "${SUMMARY_DIR}"

find_infra_log() {
  ls -1t "${RUN_LOG_DIR}"/infra-*.log 2>/dev/null | head -n1
}

INFRA_LOG="$(find_infra_log || true)"
if [[ -z "${INFRA_LOG}" ]]; then
  echo "No infra log found under ${RUN_LOG_DIR}" >&2
  exit 1
fi

find_current_step() {
  python3 - "$INFRA_LOG" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(errors="ignore")
matches = re.findall(r"'trainer/global_step':\s*([0-9]+)", text)
if matches:
    print(matches[-1])
    raise SystemExit

matches = re.findall(r"Training Step Progress:\s+.*?\s([0-9]+)/([0-9]+)\s", text)
if matches:
    print(matches[-1][0])
else:
    print(0)
PY
}

count_artifacts() {
  python3 - "$RUN_ARTIFACT_DIR" <<'PY'
from pathlib import Path
import json
import sys

run = Path(sys.argv[1])
trials = run / "trials_run"
data = {
    "trajectory_json": sum(1 for _ in trials.rglob("trajectory.json")) if trials.exists() else 0,
    "exception_txt": sum(1 for _ in trials.rglob("exception.txt")) if trials.exists() else 0,
    "result_json": sum(1 for _ in trials.rglob("result.json")) if trials.exists() else 0,
    "ckpt_dirs": sum(1 for _ in (run / "ckpts").glob("global_step_*")) if (run / "ckpts").exists() else 0,
}
print(json.dumps(data))
PY
}

collect_exception_heads() {
  python3 - "$RUN_ARTIFACT_DIR" "$SUMMARY_DIR/exception_heads.txt" <<'PY'
from pathlib import Path
import sys

run = Path(sys.argv[1]) / "trials_run"
out = Path(sys.argv[2])
files = sorted(run.rglob("exception.txt"), key=lambda p: p.stat().st_mtime, reverse=True)[:10]
with out.open("w", encoding="utf-8") as fh:
    for path in files:
        fh.write(f"FILE {path}\n")
        try:
            lines = path.read_text(errors="ignore").splitlines()
        except Exception as exc:
            fh.write(f"read_failed: {exc}\n\n")
            continue
        for line in lines[:25]:
            fh.write(f"{line}\n")
        fh.write("\n")
PY
}

collect_latest_step_metrics() {
  python3 - "$INFRA_LOG" "$SUMMARY_DIR/latest_step_metrics.txt" <<'PY'
from pathlib import Path
import re
import sys

log_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
lines = log_path.read_text(errors="ignore").splitlines()
patterns = [
    "trainer/global_step",
    "timing/step",
    "timing/wait_for_generation_buffer",
    "timing/run_training",
    "timing/sync_weights",
    "reward/avg_pass_at_4",
    "loss/avg_final_rewards",
    "async/staleness_max",
]
with out_path.open("w", encoding="utf-8") as fh:
    for line in lines[-1200:]:
        if any(p in line for p in patterns):
            fh.write(line + "\n")
PY
}

collect_gpu_snapshot() {
  local nodes="research-dev-coder-003,research-dev-coder-008,research-dev-coder-012,research-dev-coder-013,research-dev-coder-014,research-dev-coder-015"
  srun --overlap --jobid "${JOB_ID}" -w "${nodes}" -N6 -n6 bash -lc '
node=$(hostname -s)
echo "### ${node}"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits || true
echo
' > "${SUMMARY_DIR}/gpu_process_snapshot.txt" 2>&1 || true
}

stop_related_steps() {
  mapfile -t steps < <(squeue -s -j "${JOB_ID}" -h -o "%i" 2>/dev/null | grep -v "^${JOB_ID}\.batch$" || true)
  if ((${#steps[@]} > 0)); then
    printf '%s\n' "${steps[@]}" > "${SUMMARY_DIR}/steps_to_cancel.txt"
    scancel "${steps[@]}" || true
  fi
}

verify_cleanup() {
  local nodes="research-dev-coder-003,research-dev-coder-008,research-dev-coder-012,research-dev-coder-013,research-dev-coder-014,research-dev-coder-015"
  srun --overlap --jobid "${JOB_ID}" -w "${nodes}" -N6 -n6 bash -lc '
node=$(hostname -s)
echo "### ${node}"
pgrep -af "raylet|gcs_server|skyrl_entrypoint|vllm_server.py|dockerd-rootless.sh|rootlesskit|ThunderAgent|start_trainer_node_monitors|monitor_stage3_resources" || true
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits || true
echo
' > "${SUMMARY_DIR}/cleanup_verification.txt" 2>&1 || true
}

write_summary_md() {
  local artifact_json
  artifact_json="$(count_artifacts)"
  python3 - "$SUMMARY_MD" "$STATE_JSON" "$artifact_json" "$INFRA_LOG" "$SUMMARY_DIR/exception_heads.txt" "$SUMMARY_DIR/latest_step_metrics.txt" <<'PY'
from pathlib import Path
import json
import sys

summary_md = Path(sys.argv[1])
state_json = Path(sys.argv[2])
artifact_counts = json.loads(sys.argv[3])
infra_log = Path(sys.argv[4])
exception_heads = Path(sys.argv[5])
latest_metrics = Path(sys.argv[6])
state = json.loads(state_json.read_text())

lines = [
    f"# R13 Full Run Stop At Step {state['target_step']}",
    "",
    f"- run_name: `{state['run_name']}`",
    f"- stop_reason: `{state['stop_reason']}`",
    f"- observed_step: `{state['observed_step']}`",
    f"- target_step: `{state['target_step']}`",
    f"- infra_log: `{infra_log}`",
    "",
    "## Artifact Counts",
    "",
]
for key, value in artifact_counts.items():
    lines.append(f"- {key}: `{value}`")

lines.extend([
    "",
    "## Observed Exception Classes",
    "",
    "- Harbor tmux session startup failure in `harbor/agents/terminus_2/tmux_session.py:372` with `AttributeError: 'NoneType' object has no attribute 'strip'`.",
    "- Harbor / LiteLLM / hosted vLLM call failures on the `terminus_2.py -> harbor.llms.chat.py -> model.call(...)` path.",
    "- Harbor postprocessing failures where rollout details were missing: `Missing completion token ids ...` and `did not return assistant logprobs/token ids despite collect_rollout_details=True`.",
    "- Runtime `Hosted_vllmException - Server disconnected` errors surfaced during LLM interaction.",
    "",
    "## Likely Causes",
    "",
    "- The external rollout path is intermittently disconnecting under load, which matches the repeated `Hosted_vllmException - Server disconnected` errors and the missing assistant logprob/token-id details.",
    "- Harbor postprocessing assumes rollout details are always present when `collect_rollout_details=True`; once rollout returns partial metadata, the generator starts masking/erroring trajectories.",
    "- The `tmux_session.py` failure is a separate Harbor-side robustness bug: it dereferences `set_history_result.stderr` without guarding against `None`, so the real underlying tmux/setup failure is obscured.",
    "- `best_effort_release_program` warnings indicate cleanup churn on the ThunderAgent control path; they did not always kill the step, but they correlate with periods where postprocessing anomalies increased.",
    "",
    "## Latest Step Metrics Snippet",
    "",
    "```text",
    latest_metrics.read_text(errors='ignore').strip(),
    "```",
    "",
    "## Latest Exception Heads",
    "",
    "```text",
    exception_heads.read_text(errors='ignore').strip(),
    "```",
])

summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

STOP_REASON="run_exited_before_target"
OBSERVED_STEP=0

while true; do
  OBSERVED_STEP="$(find_current_step)"
  python3 - "$STATE_JSON" "$RUN_NAME" "$TARGET_STEP" "$OBSERVED_STEP" "$STOP_REASON" <<'PY'
from pathlib import Path
import json
import sys

path = Path(sys.argv[1])
payload = {
    "run_name": sys.argv[2],
    "target_step": int(sys.argv[3]),
    "observed_step": int(sys.argv[4]),
    "stop_reason": sys.argv[5],
}
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

  if (( OBSERVED_STEP >= TARGET_STEP )); then
    STOP_REASON="target_step_reached"
    break
  fi

  if ! pgrep -f "codecontest-qwen3-32b-6node-rootless-full-r13-64x4x64-stale2-conc256-20260315_0051" >/dev/null 2>&1; then
    STOP_REASON="driver_process_gone_before_target"
    break
  fi

  sleep 15
done

python3 - "$STATE_JSON" "$RUN_NAME" "$TARGET_STEP" "$OBSERVED_STEP" "$STOP_REASON" <<'PY'
from pathlib import Path
import json
import sys

path = Path(sys.argv[1])
payload = {
    "run_name": sys.argv[2],
    "target_step": int(sys.argv[3]),
    "observed_step": int(sys.argv[4]),
    "stop_reason": sys.argv[5],
}
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

collect_exception_heads
collect_latest_step_metrics
collect_gpu_snapshot
stop_related_steps
sleep 10
verify_cleanup
write_summary_md

echo "watch_complete stop_reason=${STOP_REASON} observed_step=${OBSERVED_STEP} summary=${SUMMARY_MD}"
