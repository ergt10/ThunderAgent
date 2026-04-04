#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
TOTAL_TRAINING_STEPS_RE = re.compile(r"Total training steps:\s*(\d+)")
SKYRL_ENTRYPOINT_PID_RE = re.compile(r"skyrl_entrypoint pid=(\d+)")
TRAINING_DONE_RE = re.compile(r"Training done!")
STARTED_STEP_RE = re.compile(r"Started: 'step'")
FINISHED_STEP_RE = re.compile(r"Finished: 'step'")
TRAINING_STEP_PROGRESS_RE = re.compile(r"Training Step Progress:\s+.*?\s(\d+)/(\d+)\s")
GEN_BUFFER_PROGRESS_RE = re.compile(r"Generation Buffer Progress:\s+.*?\s(\d+)/(\d+)\s")

FATAL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\*\*\* STEP .* CANCELLED AT "), "slurm_step_cancelled"),
    (re.compile(r"srun: error: .*task 0: Killed"), "srun_task_killed"),
    (re.compile(r"srun: forcing job termination"), "srun_forced_termination"),
    (re.compile(r"Traceback \(most recent call last\):"), "python_traceback"),
    (re.compile(r"Generator worker errored out with exception:"), "generator_worker_exception"),
    (re.compile(r"srun: error: .*Exited with exit code"), "srun_exit_code"),
    (re.compile(r"\bKeyboardInterrupt\b"), "keyboard_interrupt"),
    (re.compile(r"\bCancelledError\b"), "cancelled_error"),
    (re.compile(r"\bAssertionError:"), "assertion_error"),
    (re.compile(r"\bRuntimeError:"), "runtime_error"),
]


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def infer_run_name(driver_log: Path) -> str | None:
    parent = driver_log.parent
    if parent.name and parent.name != driver_log.name:
        return parent.name
    return None


def infer_artifact_dir(driver_log: Path) -> Path | None:
    run_name = infer_run_name(driver_log)
    if not run_name:
        return None
    user = getpass.getuser()
    return Path(f"/scratch/triton_cache/{user}/harbor_run_artifacts/{run_name}")


def parse_progress(pattern: re.Pattern[str], text: str) -> dict[str, int] | None:
    matches = pattern.findall(text)
    if not matches:
        return None
    cur, total = matches[-1]
    return {"current": int(cur), "total": int(total)}


def parse_trial_progress(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 2:
        return None

    header = lines[0].split("\t")
    row = lines[-1].split("\t")
    if len(row) != len(header):
        return {"raw": lines[-1]}
    return dict(zip(header, row))


def count_artifacts(artifact_dir: Path | None) -> dict[str, int] | None:
    if artifact_dir is None or not artifact_dir.exists():
        return None

    trials_dir = artifact_dir / "trials_run"
    if not trials_dir.exists():
        return None

    result_json_count = 0
    trajectory_json_count = 0
    exception_txt_count = 0
    for path in trials_dir.rglob("*"):
        if path.name == "result.json":
            result_json_count += 1
        elif path.name == "trajectory.json":
            trajectory_json_count += 1
        elif path.name == "exception.txt":
            exception_txt_count += 1

    return {
        "result_json_count": result_json_count,
        "trajectory_json_count": trajectory_json_count,
        "exception_txt_count": exception_txt_count,
    }


def find_fatal_reason(text: str) -> str | None:
    for pattern, reason in FATAL_PATTERNS:
        if pattern.search(text):
            return reason
    return None


def completed_all_steps(state: dict[str, Any]) -> bool:
    total_steps = state.get("total_training_steps")
    if total_steps is None:
        return False

    finished_steps = state.get("finished_step_count") or 0
    if finished_steps >= total_steps:
        return True

    training_progress = state.get("training_step_progress") or {}
    current = training_progress.get("current")
    total = training_progress.get("total")
    return current is not None and total is not None and current >= total_steps and total >= total_steps


def build_state(
    driver_log: Path,
    expected_total_steps: int | None,
    trial_progress_path: Path | None,
    artifact_dir: Path | None,
    include_artifacts: bool,
    check_local_pid: bool,
) -> dict[str, Any]:
    raw_text = driver_log.read_text(encoding="utf-8", errors="ignore") if driver_log.exists() else ""
    text = strip_ansi(raw_text)

    parsed_total_steps = None
    total_match = TOTAL_TRAINING_STEPS_RE.search(text)
    if total_match:
        parsed_total_steps = int(total_match.group(1))

    total_training_steps = expected_total_steps or parsed_total_steps

    pid_matches = SKYRL_ENTRYPOINT_PID_RE.findall(text)
    driver_pid = int(pid_matches[-1]) if pid_matches else None
    driver_pid_alive = None
    if driver_pid is not None and check_local_pid:
        driver_pid_alive = Path(f"/proc/{driver_pid}").exists()

    state: dict[str, Any] = {
        "driver_log": str(driver_log),
        "log_exists": driver_log.exists(),
        "log_size_bytes": driver_log.stat().st_size if driver_log.exists() else 0,
        "log_mtime_epoch": driver_log.stat().st_mtime if driver_log.exists() else None,
        "total_training_steps": total_training_steps,
        "parsed_total_training_steps": parsed_total_steps,
        "started_step_count": len(STARTED_STEP_RE.findall(text)),
        "finished_step_count": len(FINISHED_STEP_RE.findall(text)),
        "training_done": bool(TRAINING_DONE_RE.search(text)),
        "driver_pid": driver_pid,
        "driver_pid_alive": driver_pid_alive,
        "training_step_progress": parse_progress(TRAINING_STEP_PROGRESS_RE, text),
        "generation_buffer_progress": parse_progress(GEN_BUFFER_PROGRESS_RE, text),
        "trial_progress": parse_trial_progress(trial_progress_path),
        "fatal_reason": find_fatal_reason(text),
        "run_name": infer_run_name(driver_log),
        "artifact_dir": str(artifact_dir) if artifact_dir is not None else None,
    }

    if include_artifacts:
        state["artifact_counts"] = count_artifacts(artifact_dir)

    return state


def terminal_status(state: dict[str, Any], stale_timeout_sec: int, now: float) -> tuple[str, str] | None:
    if state["training_done"]:
        return ("success", "training_done_marker")

    if completed_all_steps(state):
        return ("success", "all_training_steps_finished")

    fatal_reason = state.get("fatal_reason")
    if fatal_reason:
        return ("failed", fatal_reason)

    pid_alive = state.get("driver_pid_alive")
    if state.get("driver_pid") is not None and pid_alive is False:
        total_steps = state.get("total_training_steps")
        finished_steps = state.get("finished_step_count") or 0
        if total_steps is not None and finished_steps >= total_steps:
            return ("success", "driver_exited_after_all_steps")
        return ("failed", "driver_pid_exited_without_success_marker")

    mtime = state.get("log_mtime_epoch")
    if mtime is not None and now - mtime >= stale_timeout_sec:
        total_steps = state.get("total_training_steps")
        finished_steps = state.get("finished_step_count") or 0
        if total_steps is not None and finished_steps >= total_steps:
            return ("success", "log_quiet_after_all_steps")
        if state.get("started_step_count", 0) > 0:
            return ("stalled", "log_stale_without_terminal_marker")

    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Block until a Harbor launcher_train_driver.log reaches a terminal state."
    )
    parser.add_argument("driver_log", type=Path, help="Absolute path to launcher_train_driver.log")
    parser.add_argument(
        "--expected-total-steps",
        type=int,
        default=None,
        help="Expected RL training steps. Defaults to the value parsed from the driver log.",
    )
    parser.add_argument(
        "--trial-progress",
        type=Path,
        default=None,
        help="Optional path to monitoring/trial_progress.tsv. Defaults to <log_dir>/monitoring/trial_progress.tsv.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Optional Harbor artifact root. Defaults to /scratch/triton_cache/$USER/harbor_run_artifacts/<run_name>.",
    )
    parser.add_argument(
        "--poll-sec",
        type=float,
        default=30.0,
        help="Polling interval while waiting.",
    )
    parser.add_argument(
        "--stale-timeout-sec",
        type=int,
        default=1800,
        help="If the driver log stops changing for this long after training starts, return stalled.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Inspect current state once and exit immediately.",
    )
    parser.add_argument(
        "--include-artifacts",
        action="store_true",
        help="Include result/trajectory/exception counts in the final JSON snapshot.",
    )
    parser.add_argument(
        "--check-local-pid",
        action="store_true",
        help="Opt in to checking /proc/<pid> on the current machine. Leave off if the driver runs on another node.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    driver_log = args.driver_log
    trial_progress_path = args.trial_progress
    if trial_progress_path is None:
        trial_progress_path = driver_log.parent / "monitoring" / "trial_progress.tsv"
    artifact_dir = args.artifact_dir or infer_artifact_dir(driver_log)

    while True:
        state = build_state(
            driver_log=driver_log,
            expected_total_steps=args.expected_total_steps,
            trial_progress_path=trial_progress_path,
            artifact_dir=artifact_dir,
            include_artifacts=args.include_artifacts,
            check_local_pid=args.check_local_pid,
        )
        terminal = terminal_status(state, stale_timeout_sec=args.stale_timeout_sec, now=time.time())

        if args.once:
            state["status"] = terminal[0] if terminal else "running"
            state["reason"] = terminal[1] if terminal else "awaiting_terminal_state"
            print(json.dumps(state, ensure_ascii=True))
            return 0

        if terminal is not None:
            status, reason = terminal
            state["status"] = status
            state["reason"] = reason
            if "artifact_counts" not in state:
                state["artifact_counts"] = count_artifacts(artifact_dir)
            print(json.dumps(state, ensure_ascii=True))
            return 0 if status == "success" else 2

        time.sleep(args.poll_sec)


if __name__ == "__main__":
    sys.exit(main())
