#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
LOG_TS_RE = re.compile(r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})")
TIMING_RE = re.compile(r"'timing/(?P<name>[^']+)': '?(?P<value>[0-9.]+)'?")


@dataclass
class Phase:
    name: str
    step: int
    start: pd.Timestamp
    end: pd.Timestamp
    notes: str = ""

    @property
    def duration_s(self) -> float:
        return float((self.end - self.start).total_seconds())


def read_text(path: Path) -> str:
    return path.read_bytes().replace(b"\x00", b"").decode(errors="ignore")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_monitor_tsv(path: Path) -> pd.DataFrame:
    text = read_text(path)
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return pd.DataFrame()
    header = lines[0]
    expected_tabs = header.count("\t")
    filtered_lines = [header]
    for line in lines[1:]:
        if not line[:4].isdigit():
            continue
        if line.count("\t") != expected_tabs:
            continue
        filtered_lines.append(line)
    df = pd.read_csv(pd.io.common.StringIO("\n".join(filtered_lines)), sep="\t", engine="python")
    for col in df.columns:
        if "timestamp" in col:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_many_tsvs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            continue
        df = parse_monitor_tsv(path)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def parse_launcher_events(log_path: Path) -> tuple[str, list[tuple[pd.Timestamp, str]]]:
    clean = strip_ansi(read_text(log_path))
    events: list[tuple[pd.Timestamp, str]] = []
    for line in clean.splitlines():
        m = LOG_TS_RE.search(line)
        if not m:
            continue
        ts = pd.Timestamp(datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S.%f"), tz="America/Los_Angeles")
        events.append((ts, line))
    return clean, events


def last_event_time(events: Iterable[tuple[pd.Timestamp, str]]) -> pd.Timestamp | None:
    items = list(events)
    return items[-1][0] if items else None


def find_event_ts(events: list[tuple[pd.Timestamp, str]], needle: str) -> pd.Timestamp | None:
    for ts, line in events:
        if needle in line:
            return ts
    return None


def parse_timing_metrics(clean_log: str) -> dict[str, float]:
    return {m.group("name"): float(m.group("value")) for m in TIMING_RE.finditer(clean_log)}


def ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_phase_table(events: list[tuple[pd.Timestamp, str]], clean_log: str) -> list[Phase]:
    timings = parse_timing_metrics(clean_log)
    phases: list[Phase] = []

    launcher_start = events[0][0]
    wait_start = find_event_ts(events, "Started: 'wait_for_generation_buffer'")
    wait_end = find_event_ts(events, "Finished: 'wait_for_generation_buffer'")
    step_end = find_event_ts(events, "Finished: 'step'")
    eval_start = find_event_ts(events, "Started: 'eval'")
    eval_end = find_event_ts(events, "Finished: 'eval'")
    save_ckpt_start = find_event_ts(events, "Started: 'save_checkpoints'")
    save_ckpt_end = find_event_ts(events, "Finished: 'save_checkpoints'")
    save_hf_start = find_event_ts(events, "Started: 'save_hf_model'")
    save_hf_end = find_event_ts(events, "Finished: 'save_hf_model'")

    if wait_start:
        phases.append(
            Phase(
                name="bring_up_and_preflight",
                step=0,
                start=launcher_start,
                end=wait_start,
                notes="monitor + ray/preflight + trainer bootstrap",
            )
        )

    if wait_start and wait_end:
        phases.append(
            Phase(
                name="wait_for_generation_buffer",
                step=1,
                start=wait_start,
                end=wait_end,
                notes="step=1",
            )
        )

    if wait_end and step_end:
        run_training_s = timings.get("run_training")
        if run_training_s is not None:
            run_train_end = step_end - pd.Timedelta(seconds=step_end and max((step_end - wait_end).total_seconds() - run_training_s, 0))
            # Recompute to keep the run_training segment exactly sized to the logged metric.
            run_train_start = step_end - pd.Timedelta(seconds=run_training_s) - pd.Timedelta(
                seconds=max((step_end - wait_end).total_seconds() - run_training_s, 0)
            )
            run_train_start = max(wait_end, run_train_start)
            run_train_end = run_train_start + pd.Timedelta(seconds=run_training_s)
            phases.append(
                Phase(
                    name="run_training",
                    step=1,
                    start=run_train_start,
                    end=run_train_end,
                    notes="step=1",
                )
            )
            if run_train_end < step_end:
                phases.append(
                    Phase(
                        name="step_finalize",
                        step=1,
                        start=run_train_end,
                        end=step_end,
                        notes="step=1",
                    )
                )
        else:
            phases.append(Phase(name="step", step=1, start=wait_end, end=step_end, notes="step=1"))

    if eval_start and eval_end:
        phases.append(Phase(name="eval", step=1, start=eval_start, end=eval_end, notes="20 eval trajectories"))

    if save_ckpt_start and save_ckpt_end:
        phases.append(Phase(name="save_checkpoints", step=1, start=save_ckpt_start, end=save_ckpt_end))

    if save_hf_start and save_hf_end:
        phases.append(Phase(name="save_hf_model", step=1, start=save_hf_start, end=save_hf_end))

    return phases


def format_dt(ts: pd.Timestamp | None) -> str:
    return str(ts) if ts is not None else "None"


def plot_rollout_metric(df: pd.DataFrame, value_col: str, title: str, ylabel: str, out_path: Path) -> None:
    plot_df = df.copy()
    plot_df = plot_df[plot_df["timestamp"].notna() & plot_df["source"].notna() & plot_df[value_col].notna()]
    fig, ax = plt.subplots(figsize=(18, 6), constrained_layout=True)
    for source, group in plot_df.groupby("source"):
        ax.plot(group["timestamp"], group[value_col], label=source.replace(".log", ""))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=plot_df["timestamp"].dt.tz))
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=9)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_rollout_memory(df: pd.DataFrame, out_path: Path) -> None:
    rollout_df = df[df["timestamp"].notna() & df["hostname"].str.contains("coder-008", na=False)].copy()
    rollout_df["gpu_label"] = rollout_df["hostname"].str.replace(".cloud.together.ai", "", regex=False) + ":gpu" + rollout_df["gpu_index"].astype(str)
    fig, ax = plt.subplots(figsize=(18, 6), constrained_layout=True)
    for label, group in rollout_df.groupby("gpu_label"):
        ax.plot(group["timestamp"], group["memory_used_gib"], label=label)
    ax.set_title("Rollout GPU Memory")
    ax.set_ylabel("Used GPU Memory (GiB)")
    ax.set_xlabel("Time")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=rollout_df["timestamp"].dt.tz))
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=9)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_trainer_memory(df: pd.DataFrame, out_path: Path) -> None:
    trainer_df = df[df["timestamp"].notna() & ~df["hostname"].str.contains("coder-008", na=False)].copy()
    per_node = (
        trainer_df.groupby(["timestamp", "hostname"], as_index=False)["memory_used_gib"].sum().sort_values(["hostname", "timestamp"])
    )
    fig, ax = plt.subplots(figsize=(18, 10), constrained_layout=True)
    for host, group in per_node.groupby("hostname"):
        ax.plot(group["timestamp"], group["memory_used_gib"], label=host.replace(".cloud.together.ai", ""))
    ax.set_title("Trainer GPU Memory")
    ax.set_ylabel("Aggregate Used GPU Memory (GiB)")
    ax.set_xlabel("Time")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=per_node["timestamp"].dt.tz))
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_timeline_overview(phases: list[Phase], program_df: pd.DataFrame, out_path: Path) -> None:
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(22, 9), constrained_layout=True, height_ratios=[1.2, 1.0])
    phase_colors = {
        "bring_up_and_preflight": "#5E81AC",
        "wait_for_generation_buffer": "#A3BE8C",
        "run_training": "#EBCB8B",
        "step_finalize": "#D08770",
        "eval": "#B48EAD",
        "save_checkpoints": "#88C0D0",
        "save_hf_model": "#BF616A",
    }
    for idx, phase in enumerate(phases):
        start_num = mdates.date2num(phase.start.to_pydatetime())
        width = mdates.date2num(phase.end.to_pydatetime()) - start_num
        ax0.broken_barh([(start_num, width)], (idx - 0.4, 0.8), facecolors=phase_colors.get(phase.name, "#999999"))
        ax0.text(start_num, idx, f"{phase.name} ({phase.duration_s:.1f}s)", va="center", ha="left", fontsize=9)
    ax0.set_yticks(range(len(phases)))
    ax0.set_yticklabels([f"{p.step}:{p.name}" for p in phases])
    ax0.xaxis_date()
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=phases[0].start.tz))
    ax0.set_title("Timeline Overview")
    ax0.grid(True, axis="x", alpha=0.3)

    plot_df = program_df[program_df["timestamp"].notna()].copy()
    for col, label in [
        ("total_programs", "total"),
        ("reasoning_programs", "reasoning"),
        ("acting_programs", "acting"),
        ("paused_programs", "paused"),
    ]:
        if col in plot_df.columns:
            ax1.plot(plot_df["timestamp"], plot_df[col], label=label)
    ax1.set_ylabel("Programs")
    ax1.set_xlabel("Time")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=plot_df["timestamp"].dt.tz))
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def iso_ts_to_pd(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    return pd.to_datetime(value, utc=True).tz_convert("America/Los_Angeles")


def stage_duration_s(info: dict | None) -> float:
    if not info:
        return 0.0
    start = iso_ts_to_pd(info.get("started_at"))
    end = iso_ts_to_pd(info.get("finished_at"))
    if start is None or end is None:
        return 0.0
    return float((end - start).total_seconds())


def collect_trial_rows(trials_root: Path) -> pd.DataFrame:
    rows = []
    for result_path in sorted(trials_root.glob("*/result.json")):
        try:
            obj = json.loads(result_path.read_text())
        except Exception:
            continue
        started = iso_ts_to_pd(obj.get("started_at"))
        finished = iso_ts_to_pd(obj.get("finished_at"))
        if started is None or finished is None:
            continue
        total_s = float((finished - started).total_seconds())
        env_s = stage_duration_s(obj.get("environment_setup"))
        agent_setup_s = stage_duration_s(obj.get("agent_setup"))
        agent_exec_s = stage_duration_s(obj.get("agent_execution"))
        verifier_s = stage_duration_s(obj.get("verifier"))
        known = env_s + agent_setup_s + agent_exec_s + verifier_s
        post_other_s = max(total_s - known, 0.0)
        exc = obj.get("exception_info")
        rows.append(
            {
                "trial_name": obj.get("trial_name"),
                "task_name": obj.get("task_name"),
                "total_s": total_s,
                "environment_setup_s": env_s,
                "agent_setup_s": agent_setup_s,
                "agent_execution_s": agent_exec_s,
                "verifier_s": verifier_s,
                "post_trial_other_s": post_other_s,
                "successful": exc is None,
                "exception_type": (exc or {}).get("exception_type"),
            }
        )
    return pd.DataFrame(rows)


def summarize_trial_groups(trial_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    groups = {
        "All Trials": trial_df,
        "Successful Trials": trial_df[trial_df["successful"]],
        "Failed Trials": trial_df[~trial_df["successful"]],
    }
    out = {}
    for name, group in groups.items():
        if group.empty:
            continue
        out[name] = {
            "trials_count": int(len(group)),
            "avg_total_sec": float(group["total_s"].mean()),
            "environment_setup": float(group["environment_setup_s"].mean()),
            "agent_setup": float(group["agent_setup_s"].mean()),
            "agent_execution": float(group["agent_execution_s"].mean()),
            "verifier": float(group["verifier_s"].mean()),
            "post_trial_other": float(group["post_trial_other_s"].mean()),
        }
    return out


def plot_trial_stage_breakdown(summary: dict[str, dict[str, float]], out_path: Path) -> None:
    labels = list(summary.keys())
    phase_cols = [
        ("environment_setup", "#5E81AC"),
        ("agent_setup", "#A3BE8C"),
        ("agent_execution", "#EBCB8B"),
        ("verifier", "#D08770"),
        ("post_trial_other", "#B48EAD"),
    ]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(15, 9), constrained_layout=True)
    bottom = np.zeros(len(labels))
    for phase, color in phase_cols:
        vals = np.array([summary[label][phase] for label in labels], dtype=float)
        ax.bar(x, vals, bottom=bottom, label=phase, color=color)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Average Seconds per Trial")
    ax.set_title("Harbor Trial Stage Breakdown")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_timeline_report(
    run_name: str,
    clean_log: str,
    events: list[tuple[pd.Timestamp, str]],
    phases: list[Phase],
    gpu_df: pd.DataFrame,
    vllm_df: pd.DataFrame,
) -> str:
    launcher_start = events[0][0] if events else None
    launcher_end = find_event_ts(events, "Training done!") or last_event_time(events)
    timing_metrics = parse_timing_metrics(clean_log)
    agent_timeout_count = clean_log.count("agent_timeout")
    thunder_wait_timeout_count = clean_log.count("thunder_wait_timeout")
    none_type_count = clean_log.count("NoneType")

    step_rows = []
    wait_s = timing_metrics.get("wait_for_generation_buffer", 0.0)
    train_s = timing_metrics.get("run_training", 0.0)
    step_s = timing_metrics.get("step", 0.0)
    finalize_s = max(step_s - wait_s - train_s, 0.0)
    step_rows.append((1, wait_s, train_s, finalize_s, 1))

    trainer_df = gpu_df[~gpu_df["hostname"].str.contains("coder-008", na=False)].copy()
    trainer_peaks = (
        trainer_df.assign(
            trainer_gpu=trainer_df["hostname"].str.replace(".cloud.together.ai", "", regex=False)
            + ":gpu"
            + trainer_df["gpu_index"].astype(str)
        )
        .groupby("trainer_gpu", as_index=False)["memory_used_gib"]
        .max()
        .sort_values("memory_used_gib", ascending=False)
    )
    trainer_node_peaks = (
        trainer_df.groupby(["timestamp", "hostname"], as_index=False)["memory_used_gib"]
        .sum()
        .groupby("hostname", as_index=False)["memory_used_gib"]
        .max()
        .sort_values("memory_used_gib", ascending=False)
    )

    rollout_gpu_df = gpu_df[gpu_df["hostname"].str.contains("coder-008", na=False)].copy()
    rollout_gpu_peaks = (
        rollout_gpu_df.assign(
            rollout_gpu=rollout_gpu_df["hostname"].str.replace(".cloud.together.ai", "", regex=False)
            + ":gpu"
            + rollout_gpu_df["gpu_index"].astype(str)
        )
        .groupby("rollout_gpu", as_index=False)["memory_used_gib"]
        .max()
        .sort_values("memory_used_gib", ascending=False)
    )
    rollout_peaks = {
        "peak_kv_usage_pct": vllm_df.groupby("source", as_index=False)["kv_cache_usage_pct"].max().sort_values(
            "kv_cache_usage_pct", ascending=False
        ),
        "peak_prefix_cache_hit_rate_pct": vllm_df.groupby("source", as_index=False)["prefix_cache_hit_rate_pct"].max().sort_values(
            "prefix_cache_hit_rate_pct", ascending=False
        ),
        "peak_num_requests_running": vllm_df.groupby("source", as_index=False)["num_requests_running"].max().sort_values(
            "num_requests_running", ascending=False
        ),
        "peak_num_requests_waiting": vllm_df.groupby("source", as_index=False)["num_requests_waiting"].max().sort_values(
            "num_requests_waiting", ascending=False
        ),
    }

    lines = [f"# Full Run Timeline Analysis: {run_name}", "", "## Window", ""]
    lines += [
        f"- launcher_start: {format_dt(launcher_start)}",
        f"- launcher_end: {format_dt(launcher_end)}",
        "- cancel_time: None",
        f"- agent_timeout_count: {agent_timeout_count}",
        f"- thunder_wait_timeout_count: {thunder_wait_timeout_count}",
        f"- none_type_trajectory_id_count: {none_type_count}",
        "",
        "## Phase Table",
        "",
        "| phase | step | start | end | duration_s | notes |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for phase in phases:
        lines.append(
            f"| {phase.name} | {phase.step} | {phase.start} | {phase.end} | {phase.duration_s:.2f} | {phase.notes} |"
        )

    lines += ["", "## Step Summary", "", "| step | wait_s | train_s | finalize_s | global_step_after |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in step_rows:
        lines.append(f"| {row[0]} | {row[1]:.2f} | {row[2]:.2f} | {row[3]:.2f} | {row[4]} |")

    lines += ["", "## Trainer Peaks", "", "| trainer_gpu | peak_reserved_gib |", "| --- | ---: |"]
    for _, row in trainer_peaks.iterrows():
        lines.append(f"| {row['trainer_gpu']} | {row['memory_used_gib']:.2f} |")
    lines += ["", "| trainer_node | peak_reserved_gib |", "| --- | ---: |"]
    for _, row in trainer_node_peaks.iterrows():
        lines.append(f"| {row['hostname'].replace('.cloud.together.ai', '')} | {row['memory_used_gib']:.2f} |")

    lines += ["", "## Rollout Peaks", "", "| rollout_gpu | peak_used_gib |", "| --- | ---: |"]
    for _, row in rollout_gpu_peaks.iterrows():
        lines.append(f"| {row['rollout_gpu']} | {row['memory_used_gib']:.2f} |")

    for title, frame in rollout_peaks.items():
        pretty = title.replace("_", " ")
        lines += ["", f"| rollout_backend | {pretty} |", "| --- | ---: |"]
        metric_col = frame.columns[-1]
        for _, row in frame.iterrows():
            lines.append(f"| {row['source'].replace('.log', '')} | {row[metric_col]:.2f} |")

    return "\n".join(lines) + "\n"


def build_trial_stage_report(run_name: str, trial_df: pd.DataFrame, summary: dict[str, dict[str, float]]) -> str:
    exception_counts = Counter(x for x in trial_df["exception_type"].dropna())
    lines = [
        "# Harbor Trial Stage Breakdown",
        "",
        "## Notes",
        "",
        "- This run does not expose `agent_result.metadata.api_request_times_msec`, so the report stays at top-level trial phases.",
        "- `environment_setup / agent_setup / agent_execution / verifier` come from `result.json` timing blocks.",
        "- `post_trial_other` is the residual: `total - known_phase_sum`.",
    ]
    if exception_counts:
        lines += ["", "## Exception Types", ""]
        for name, count in exception_counts.most_common():
            lines.append(f"- {name}: {count}")
    for section, values in summary.items():
        lines += [
            "",
            f"## {section}",
            "",
            f"- trials_count: {values['trials_count']}",
            f"- avg_total_sec: {values['avg_total_sec']:.2f}",
            f"- environment_setup: {values['environment_setup']:.2f}s ({values['environment_setup'] / values['avg_total_sec'] * 100:.2f}%)",
            f"- agent_setup: {values['agent_setup']:.2f}s ({values['agent_setup'] / values['avg_total_sec'] * 100:.2f}%)",
            f"- agent_execution: {values['agent_execution']:.2f}s ({values['agent_execution'] / values['avg_total_sec'] * 100:.2f}%)",
            f"- verifier: {values['verifier']:.2f}s ({values['verifier'] / values['avg_total_sec'] * 100:.2f}%)",
            f"- post_trial_other: {values['post_trial_other']:.2f}s ({values['post_trial_other'] / values['avg_total_sec'] * 100:.2f}%)",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--trials-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--trainer-monitors-dir", type=Path, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    monitoring = args.log_dir / "monitoring"
    launcher_log = args.log_dir / "launcher_train_driver.log"

    clean_log, events = parse_launcher_events(launcher_log)
    phases = build_phase_table(events, clean_log)

    vllm_df = parse_monitor_tsv(monitoring / "vllm_metrics.tsv")
    vllm_df = ensure_numeric(
        vllm_df,
        [
            "kv_cache_usage_pct",
            "prompt_throughput_tokens_per_s",
            "generation_throughput_tokens_per_s",
            "num_requests_running",
            "num_requests_waiting",
            "prefix_cache_hit_rate_pct",
        ],
    )
    vllm_df = vllm_df[vllm_df["metrics_url"].astype(str).str.contains("/metrics", na=False)]

    gpu_paths = [monitoring / "gpu_summary.tsv"]
    if args.trainer_monitors_dir and args.trainer_monitors_dir.exists():
        gpu_paths.extend(sorted(args.trainer_monitors_dir.glob("*/gpu_summary.tsv")))
    gpu_df = load_many_tsvs(gpu_paths)
    gpu_df = ensure_numeric(gpu_df, ["memory_total_mib", "memory_used_mib", "memory_free_mib", "util_gpu_pct", "util_mem_pct"])
    gpu_df["memory_used_gib"] = gpu_df["memory_used_mib"] / 1024.0

    program_df = parse_monitor_tsv(monitoring / "thunderagent_program_state.tsv")
    program_df = ensure_numeric(
        program_df,
        ["total_programs", "reasoning_programs", "acting_programs", "paused_programs", "created_total", "released_total"],
    )
    program_df = program_df[program_df["timestamp"].notna() & program_df["total_programs"].notna()]

    trial_df = collect_trial_rows(args.trials_root)
    summary = summarize_trial_groups(trial_df)

    plot_rollout_metric(
        vllm_df,
        "kv_cache_usage_pct",
        "Rollout KV Cache Usage",
        "KV Cache Usage (%)",
        args.output_dir / "rollout_kv_usage.png",
    )
    plot_rollout_memory(gpu_df, args.output_dir / "rollout_memory.png")
    plot_rollout_metric(
        vllm_df,
        "num_requests_running",
        "Rollout Running Requests",
        "Requests Running",
        args.output_dir / "rollout_num_requests_running.png",
    )
    plot_rollout_metric(
        vllm_df,
        "num_requests_waiting",
        "Rollout Waiting Requests",
        "Requests Waiting",
        args.output_dir / "rollout_num_requests_waiting.png",
    )
    plot_rollout_metric(
        vllm_df,
        "prefix_cache_hit_rate_pct",
        "Rollout Prefix Cache Hit Rate",
        "Prefix Cache Hit Rate (%)",
        args.output_dir / "rollout_prefix_cache_hit_rate.png",
    )
    plot_trainer_memory(gpu_df, args.output_dir / "trainer_memory.png")
    plot_timeline_overview(phases, program_df, args.output_dir / "timeline_overview.png")
    plot_trial_stage_breakdown(summary, args.output_dir / "trial_stage_breakdown.png")

    (args.output_dir / "timeline_report.md").write_text(build_timeline_report(args.run_name, clean_log, events, phases, gpu_df, vllm_df))
    (args.output_dir / "trial_stage_breakdown_report.md").write_text(
        build_trial_stage_report(args.run_name, trial_df, summary)
    )


if __name__ == "__main__":
    main()
