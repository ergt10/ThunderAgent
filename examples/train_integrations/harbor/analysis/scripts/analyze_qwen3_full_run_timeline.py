#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import pandas as pd


ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
LAUNCHER_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[\.,]\d{3})")
GLOBAL_STEP_RE = re.compile(r"trainer/global_step':\s*(\d+)")
CANCELLED_AT_RE = re.compile(r"CANCELLED AT (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")
PREFIX_CACHE_HIT_RATE_RE = re.compile(r"Prefix cache hit rate:\s*([0-9.]+)%")
VLLM_RAW_TS_RE = re.compile(r"INFO (\d{2})-(\d{2}) (\d{2}:\d{2}:\d{2})")
GIB = 1024**3


@dataclass
class StepInfo:
    step_index: int
    step_start: pd.Timestamp
    wait_start: Optional[pd.Timestamp] = None
    wait_end: Optional[pd.Timestamp] = None
    train_start: Optional[pd.Timestamp] = None
    train_end: Optional[pd.Timestamp] = None
    step_end: Optional[pd.Timestamp] = None
    global_step_after: Optional[int] = None


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_launcher_ts(line: str, tz: ZoneInfo) -> Optional[pd.Timestamp]:
    match = LAUNCHER_TS_RE.search(line)
    if not match:
        return None
    return pd.Timestamp(match.group(1).replace(",", ".")).tz_localize(tz)


def parse_iso_ts(ts: str, tz: ZoneInfo) -> pd.Timestamp:
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(tz)
    return stamp


def parse_tsv_timestamp_bounds(path: Path, tz: ZoneInfo) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    if not path.exists():
        return None, None
    try:
        df = pd.read_csv(path, sep="\t")
    except Exception:
        return None, None
    ts_col = None
    for candidate in ("event_timestamp", "timestamp"):
        if candidate in df.columns:
            ts_col = candidate
            break
    if ts_col is None:
        return None, None
    series = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    series = series.dropna()
    if series.empty:
        return None, None
    localized = series.dt.tz_convert(tz)
    return localized.min(), localized.max()


def infer_monitor_window(run_dir: Path, tz: ZoneInfo) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    candidates = [
        run_dir / "monitoring" / "trial_progress.tsv",
        run_dir / "rollout" / "monitoring" / "vllm_metrics.tsv",
        run_dir / "rollout" / "monitoring" / "gpu_summary.tsv",
        run_dir / "monitoring" / "thunderagent_backend_state.tsv",
        run_dir / "monitoring" / "thunderagent_program_state.tsv",
    ]
    mins: List[pd.Timestamp] = []
    maxs: List[pd.Timestamp] = []
    for candidate in candidates:
        lower, upper = parse_tsv_timestamp_bounds(candidate, tz)
        if lower is not None:
            mins.append(lower)
        if upper is not None:
            maxs.append(upper)
    if not mins or not maxs:
        return None, None
    return min(mins), max(maxs)


def iter_clean_lines(path: Path) -> Iterable[str]:
    with path.open("r", errors="replace") as handle:
        for raw_line in handle:
            yield strip_ansi(raw_line.rstrip("\n"))


def parse_launcher(
    launcher_path: Path,
    tz: ZoneInfo,
) -> Dict[str, object]:
    steps: List[StepInfo] = []
    current_step: Optional[StepInfo] = None
    first_ts: Optional[pd.Timestamp] = None
    last_ts: Optional[pd.Timestamp] = None
    agent_timeout_times: List[pd.Timestamp] = []
    none_error_times: List[pd.Timestamp] = []
    cancel_time: Optional[pd.Timestamp] = None

    for line in iter_clean_lines(launcher_path):
        ts = parse_launcher_ts(line, tz)
        if ts is not None:
            first_ts = ts if first_ts is None else min(first_ts, ts)
            last_ts = ts if last_ts is None else max(last_ts, ts)

        cancel_match = CANCELLED_AT_RE.search(line)
        if cancel_match:
            cancel_time = parse_iso_ts(cancel_match.group(1), tz)

        if "failed (stop_reason=agent_timeout)" in line and ts is not None:
            agent_timeout_times.append(ts)
        if "Generator worker errored out with exception: 'NoneType' object has no attribute 'trajectory_id'" in line:
            if ts is not None:
                none_error_times.append(ts)

        if "Started: 'step'" in line and ts is not None:
            current_step = StepInfo(step_index=len(steps) + 1, step_start=ts)
        elif "Started: 'wait_for_generation_buffer'" in line and ts is not None and current_step is not None:
            current_step.wait_start = ts
        elif "Finished: 'wait_for_generation_buffer'" in line and ts is not None and current_step is not None:
            current_step.wait_end = ts
        elif "Started: 'run_training'" in line and ts is not None and current_step is not None:
            current_step.train_start = ts
        elif "Finished: 'run_training'" in line and ts is not None and current_step is not None:
            current_step.train_end = ts
        elif "Finished: 'step'" in line and ts is not None and current_step is not None:
            current_step.step_end = ts
            steps.append(current_step)
            current_step = None
        else:
            gs_match = GLOBAL_STEP_RE.search(line)
            if gs_match and steps:
                steps[-1].global_step_after = int(gs_match.group(1))

    if current_step is not None:
        steps.append(current_step)

    return {
        "first_ts": first_ts,
        "last_ts": last_ts,
        "steps": steps,
        "agent_timeout_times": agent_timeout_times,
        "none_error_times": none_error_times,
        "cancel_time": cancel_time,
    }


def parse_thunder_wait_timeout_times(thunder_path: Path, tz: ZoneInfo) -> List[pd.Timestamp]:
    times: List[pd.Timestamp] = []
    if not thunder_path.exists():
        return times
    for line in iter_clean_lines(thunder_path):
        if "wait timeout after " not in line:
            continue
        try:
            prefix = line.split(" | ", 1)[0]
            times.append(pd.Timestamp(prefix).tz_localize(tz))
        except Exception:
            continue
    return times


def coerce_numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="Float64")
    return pd.to_numeric(df[column], errors="coerce")


def parse_legacy_vllm_event_time(raw_line: object, sample_time: pd.Timestamp) -> Optional[pd.Timestamp]:
    if pd.isna(raw_line):
        return None
    text = strip_ansi(str(raw_line))
    match = VLLM_RAW_TS_RE.search(text)
    if not match:
        return None
    month, day, hms = match.groups()
    try:
        stamp = pd.Timestamp(f"{sample_time.year}-{month}-{day} {hms}")
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize(sample_time.tz)
        return stamp
    except Exception:
        return None


def load_rollout_vllm_metrics(vllm_metrics_path: Path, tz: ZoneInfo) -> pd.DataFrame:
    df = pd.read_csv(vllm_metrics_path, sep="\t")
    if df.empty:
        raise RuntimeError("No rollout rows found in vllm_metrics.tsv")

    sample_time = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_convert(tz)
    event_time = None
    if "event_timestamp" in df.columns:
        event_time = pd.to_datetime(df["event_timestamp"], utc=True, errors="coerce").dt.tz_convert(tz)

    raw_lines = df["raw_line"] if "raw_line" in df.columns else pd.Series([""] * len(df), index=df.index)
    legacy_times = [
        parse_legacy_vllm_event_time(raw_line, sample_ts)
        for raw_line, sample_ts in zip(raw_lines, sample_time)
    ]
    legacy_time_series = pd.Series(legacy_times, index=df.index, dtype="object")
    if event_time is not None:
        resolved_time = event_time.where(event_time.notna(), legacy_time_series)
    else:
        resolved_time = legacy_time_series
    resolved_time = pd.Series(resolved_time, index=df.index)
    df["time"] = resolved_time.where(resolved_time.notna(), sample_time)
    df["sample_time"] = sample_time
    for column in (
        "kv_cache_usage_pct",
        "kv_cache_size_tokens",
        "prompt_throughput_tokens_per_s",
        "generation_throughput_tokens_per_s",
        "num_requests_running",
        "num_requests_waiting",
        "prefix_cache_hit_rate_pct",
        "prefix_cache_queries_total",
        "prefix_cache_hits_total",
        "prompt_tokens_total",
        "generation_tokens_total",
    ):
        df[column] = coerce_numeric_column(df, column)
    if "kv_cache_usage_pct" in df.columns and df["kv_cache_usage_pct"].notna().any():
        # Older monitors persisted the raw 0-1 ratio from vLLM even though the
        # column name says "_pct". Normalize those historical TSVs to 0-100.
        if float(df["kv_cache_usage_pct"].max()) <= 1.0:
            df["kv_cache_usage_pct"] = df["kv_cache_usage_pct"] * 100.0
    return df


def build_phase_rows(
    steps: List[StepInfo],
    launcher_start: pd.Timestamp,
    final_end: pd.Timestamp,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if steps and launcher_start < steps[0].step_start:
        rows.append(
            {
                "phase": "bring_up_and_preflight",
                "step_index": 0,
                "start": launcher_start,
                "end": steps[0].step_start,
                "duration_s": (steps[0].step_start - launcher_start).total_seconds(),
                "notes": "monitor + ray/preflight + trainer bootstrap",
            }
        )

    for step in steps:
        if step.wait_start is not None and step.wait_end is not None:
            rows.append(
                {
                    "phase": "wait_for_generation_buffer",
                    "step_index": step.step_index,
                    "start": step.wait_start,
                    "end": step.wait_end,
                    "duration_s": (step.wait_end - step.wait_start).total_seconds(),
                    "notes": f"step={step.step_index}",
                }
            )
        if step.train_start is not None and step.train_end is not None:
            rows.append(
                {
                    "phase": "run_training",
                    "step_index": step.step_index,
                    "start": step.train_start,
                    "end": step.train_end,
                    "duration_s": (step.train_end - step.train_start).total_seconds(),
                    "notes": f"step={step.step_index}",
                }
            )
        finalize_start = step.train_end or step.wait_end
        finalize_end = step.step_end
        if finalize_start is not None and finalize_end is not None and finalize_end > finalize_start:
            rows.append(
                {
                    "phase": "step_finalize",
                    "step_index": step.step_index,
                    "start": finalize_start,
                    "end": finalize_end,
                    "duration_s": (finalize_end - finalize_start).total_seconds(),
                    "notes": f"step={step.step_index}",
                }
            )

    if steps:
        last_step = steps[-1]
        if last_step.step_end is None:
            active_phase = None
            active_start = None
            if last_step.train_start is not None and last_step.train_end is None:
                active_phase = "run_training_active"
                active_start = last_step.train_start
            elif last_step.wait_start is not None and last_step.wait_end is None:
                active_phase = "wait_for_generation_buffer_active"
                active_start = last_step.wait_start
            elif last_step.train_end is not None:
                active_phase = "step_finalize_active"
                active_start = last_step.train_end
            elif last_step.wait_end is not None:
                active_phase = "step_finalize_active"
                active_start = last_step.wait_end

            if active_phase is not None and active_start is not None and final_end > active_start:
                rows.append(
                    {
                        "phase": active_phase,
                        "step_index": last_step.step_index,
                        "start": active_start,
                        "end": final_end,
                        "duration_s": (final_end - active_start).total_seconds(),
                        "notes": f"step={last_step.step_index} incomplete",
                    }
                )
    return rows


def parse_trainer_memory(
    memory_events_dir: Path,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> Dict[str, pd.DataFrame]:
    records: List[Dict[str, object]] = []
    for path in sorted(memory_events_dir.glob("*.jsonl")):
        with path.open("r", errors="replace") as handle:
            for raw_line in handle:
                obj = json.loads(raw_line)
                ts = parse_iso_ts(obj["timestamp"], start_time.tzinfo)  # type: ignore[arg-type]
                host = obj["hostname"].replace(".cloud.together.ai", "")
                gpu = int(obj["gpu_id"])
                worker_role = obj["worker_role"]
                role = "policy" if "Policy" in worker_role else "ref"
                worker_key = f"{host}|gpu{gpu}|{role}|rank{obj['rank']}"
                records.append(
                    {
                        "time": ts,
                        "worker_key": worker_key,
                        "host": host,
                        "gpu_label": f"{host}:gpu{gpu}",
                        "reserved_gib": obj["metrics"]["cuda_reserved_bytes"] / GIB,
                        "allocated_gib": obj["metrics"]["cuda_allocated_bytes"] / GIB,
                    }
                )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise RuntimeError("No trainer memory events found")

    sample_index = pd.date_range(
        start=start_time.floor("10s"),
        end=end_time.ceil("10s"),
        freq="10s",
        tz=start_time.tz,
    )

    worker_meta = df[["worker_key", "gpu_label", "host"]].drop_duplicates().set_index("worker_key")

    physical_frames: Dict[str, pd.DataFrame] = {}
    for metric in ("reserved_gib", "allocated_gib"):
        pivot = df.pivot_table(index="time", columns="worker_key", values=metric, aggfunc="last").sort_index()
        pivot = pivot.reindex(pivot.index.union(sample_index)).sort_index().ffill()
        pivot = pivot.reindex(sample_index).fillna(0.0)
        physical = pivot.T.groupby(worker_meta["gpu_label"]).sum().T
        physical = physical.reindex(sorted(physical.columns), axis=1)
        physical_frames[metric] = physical

    node_reserved = physical_frames["reserved_gib"].T.groupby(lambda col: col.split(":gpu")[0]).sum().T
    node_allocated = physical_frames["allocated_gib"].T.groupby(lambda col: col.split(":gpu")[0]).sum().T

    return {
        "gpu_reserved": physical_frames["reserved_gib"],
        "gpu_allocated": physical_frames["allocated_gib"],
        "node_reserved": node_reserved,
        "node_allocated": node_allocated,
    }


def parse_rollout_gpu_summary(
    gpu_summary_path: Path,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> pd.DataFrame:
    df = pd.read_csv(gpu_summary_path, sep="\t")
    df["time"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(start_time.tz)
    df["gpu_label"] = df["hostname"].str.replace(".cloud.together.ai", "", regex=False) + ":gpu" + df[
        "gpu_index"
    ].astype(str)
    pivot = df.pivot_table(index="time", columns="gpu_label", values="memory_used_mib", aggfunc="last").sort_index()
    sample_index = pd.date_range(
        start=start_time.floor("10s"),
        end=end_time.ceil("10s"),
        freq="10s",
        tz=start_time.tz,
    )
    pivot = pivot.reindex(pivot.index.union(sample_index)).sort_index().ffill()
    pivot = pivot.reindex(sample_index).fillna(0.0) / 1024.0
    return pivot.reindex(sorted(pivot.columns), axis=1)


def parse_gpu_total_gib_from_summary(gpu_summary_path: Path) -> float:
    df = pd.read_csv(gpu_summary_path, sep="\t", usecols=["memory_total_mib"])
    if df.empty:
        raise RuntimeError("No GPU total memory rows found in gpu_summary.tsv")
    return float(df["memory_total_mib"].max()) / 1024.0


def parse_rollout_kv_usage(
    vllm_metrics_path: Path,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> pd.DataFrame:
    df = load_rollout_vllm_metrics(vllm_metrics_path, start_time.tz)  # type: ignore[arg-type]
    df = df[df["kv_cache_usage_pct"].notna()].copy()
    if df.empty:
        raise RuntimeError("No rollout KV usage rows found in vllm_metrics.tsv")

    pivot = (
        df.pivot_table(index="time", columns="source", values="kv_cache_usage_pct", aggfunc="last")
        .sort_index()
        .rename(columns=lambda value: str(value).replace(".log", ""))
    )
    sample_index = pd.date_range(
        start=start_time.floor("10s"),
        end=end_time.ceil("10s"),
        freq="10s",
        tz=start_time.tz,
    )
    pivot = pivot.reindex(pivot.index.union(sample_index)).sort_index().ffill()
    pivot = pivot.reindex(sample_index).fillna(0.0)
    return pivot.reindex(sorted(pivot.columns), axis=1)


def parse_rollout_prefix_cache_hit_rate(
    vllm_metrics_path: Path,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> pd.DataFrame:
    df = load_rollout_vllm_metrics(vllm_metrics_path, start_time.tz)  # type: ignore[arg-type]
    if "prefix_cache_hit_rate_pct" not in df.columns or df["prefix_cache_hit_rate_pct"].isna().all():
        prefix_pct = df["raw_line"].astype(str).str.extract(PREFIX_CACHE_HIT_RATE_RE, expand=False)
        df = df.assign(prefix_cache_hit_rate_pct=pd.to_numeric(prefix_pct, errors="coerce"))
    df = df[df["prefix_cache_hit_rate_pct"].notna()].copy()
    if df.empty:
        raise RuntimeError("No rollout prefix cache hit rate rows found in vllm_metrics.tsv")

    pivot = (
        df.pivot_table(index="time", columns="source", values="prefix_cache_hit_rate_pct", aggfunc="last")
        .sort_index()
        .rename(columns=lambda value: str(value).replace(".log", ""))
    )
    sample_index = pd.date_range(
        start=start_time.floor("10s"),
        end=end_time.ceil("10s"),
        freq="10s",
        tz=start_time.tz,
    )
    pivot = pivot.reindex(pivot.index.union(sample_index)).sort_index().ffill()
    pivot = pivot.reindex(sample_index).fillna(0.0)
    return pivot.reindex(sorted(pivot.columns), axis=1)


def parse_rollout_num_requests_running(
    vllm_metrics_path: Path,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> pd.DataFrame:
    df = load_rollout_vllm_metrics(vllm_metrics_path, start_time.tz)  # type: ignore[arg-type]
    df = df[df["num_requests_running"].notna()].copy()
    if df.empty:
        raise RuntimeError("No rollout num_requests_running rows found in vllm_metrics.tsv")

    pivot = (
        df.pivot_table(index="time", columns="source", values="num_requests_running", aggfunc="last")
        .sort_index()
        .rename(columns=lambda value: str(value).replace(".log", ""))
    )
    sample_index = pd.date_range(
        start=start_time.floor("10s"),
        end=end_time.ceil("10s"),
        freq="10s",
        tz=start_time.tz,
    )
    pivot = pivot.reindex(pivot.index.union(sample_index)).sort_index().ffill()
    pivot = pivot.reindex(sample_index).fillna(0.0)
    return pivot.reindex(sorted(pivot.columns), axis=1)


def parse_rollout_num_requests_waiting(
    vllm_metrics_path: Path,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> pd.DataFrame:
    df = load_rollout_vllm_metrics(vllm_metrics_path, start_time.tz)  # type: ignore[arg-type]
    df = df[df["num_requests_waiting"].notna()].copy()
    if df.empty:
        raise RuntimeError("No rollout num_requests_waiting rows found in vllm_metrics.tsv")

    pivot = (
        df.pivot_table(index="time", columns="source", values="num_requests_waiting", aggfunc="last")
        .sort_index()
        .rename(columns=lambda value: str(value).replace(".log", ""))
    )
    sample_index = pd.date_range(
        start=start_time.floor("10s"),
        end=end_time.ceil("10s"),
        freq="10s",
        tz=start_time.tz,
    )
    pivot = pivot.reindex(pivot.index.union(sample_index)).sort_index().ffill()
    pivot = pivot.reindex(sample_index).fillna(0.0)
    return pivot.reindex(sorted(pivot.columns), axis=1)


def make_timeline_plot(
    output_path: Path,
    phase_rows: List[Dict[str, object]],
    agent_timeout_times: List[pd.Timestamp],
    thunder_wait_timeout_times: List[pd.Timestamp],
    none_error_times: List[pd.Timestamp],
    cancel_time: Optional[pd.Timestamp],
) -> None:
    if not phase_rows:
        raise RuntimeError("No phase rows available for timeline plot")

    tz = phase_rows[0]["start"].tzinfo  # type: ignore[assignment]
    phase_style = {
        "bring_up_and_preflight": {"label": "Bring-up / Preflight", "color": "#7f8c8d", "hatch": None},
        "observed_run_span": {"label": "Observed Run Span", "color": "#9c755f", "hatch": ".."},
        "wait_for_generation_buffer": {"label": "Wait Buffer", "color": "#4c78a8", "hatch": None},
        "wait_for_generation_buffer_active": {"label": "Wait Buffer (active)", "color": "#4c78a8", "hatch": "////"},
        "run_training": {"label": "Run Training", "color": "#f58518", "hatch": None},
        "run_training_active": {"label": "Run Training (active)", "color": "#f58518", "hatch": "////"},
        "step_finalize": {"label": "Step Finalize", "color": "#54a24b", "hatch": None},
        "step_finalize_active": {"label": "Step Finalize (active)", "color": "#54a24b", "hatch": "////"},
    }

    step_indices = sorted({int(row["step_index"]) for row in phase_rows if int(row["step_index"]) > 0})
    lane_labels = ["Bring-up"] + [f"Step {step}" for step in step_indices]
    lane_positions = {0: len(lane_labels) - 1}
    for offset, step in enumerate(step_indices, start=1):
        lane_positions[step] = len(lane_labels) - 1 - offset

    fig = plt.figure(figsize=(22, 9))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[4.6, 2.2],
        height_ratios=[3.2, 1.2],
        hspace=0.06,
        wspace=0.05,
    )
    ax_top = fig.add_subplot(grid[0, 0])
    ax_side = fig.add_subplot(grid[0, 1])
    ax_bottom = fig.add_subplot(grid[1, :], sharex=ax_top)

    x_starts: List[float] = []
    x_ends: List[float] = []
    legend_seen = set()
    for row in sorted(phase_rows, key=lambda item: (int(item["step_index"]), item["start"])):
        phase_name = str(row["phase"])
        style = phase_style[phase_name]
        lane_y = lane_positions[int(row["step_index"])]
        start_num = mdates.date2num(row["start"])  # type: ignore[arg-type]
        end_num = mdates.date2num(row["end"])  # type: ignore[arg-type]
        duration_num = end_num - start_num
        x_starts.append(start_num)
        x_ends.append(end_num)

        label = style["label"] if phase_name not in legend_seen else None
        ax_top.barh(
            lane_y,
            duration_num,
            left=start_num,
            height=0.72,
            color=style["color"],
            edgecolor="#202020",
            linewidth=0.8,
            hatch=style["hatch"],
            alpha=0.95,
            label=label,
        )
        legend_seen.add(phase_name)

    if cancel_time is not None:
        cancel_num = mdates.date2num(cancel_time)
        x_starts.append(cancel_num)
        x_ends.append(cancel_num)
        ax_top.axvline(cancel_num, color="black", linestyle="-", linewidth=1.5)
        ax_bottom.axvline(cancel_num, color="black", linestyle="-", linewidth=1.5)
        ax_top.text(cancel_num, max(lane_positions.values()) + 0.7, "cancel", rotation=90, va="bottom", ha="center", fontsize=8)

    def counts_per_minute(times: List[pd.Timestamp]) -> pd.Series:
        if not times:
            return pd.Series(dtype=float)
        series = pd.Series(1, index=pd.DatetimeIndex(times)).sort_index()
        return series.resample("1min").sum()

    event_specs = [
        ("agent_timeout", counts_per_minute(agent_timeout_times), "#b279a2"),
        ("thunder wait timeout", counts_per_minute(thunder_wait_timeout_times), "#9d755d"),
        ("NoneType trajectory_id", counts_per_minute(none_error_times), "#d62728"),
    ]
    for label, series, color in event_specs:
        if series.empty:
            continue
        ax_bottom.plot(series.index, series.values, drawstyle="steps-mid", linewidth=2.0, color=color, label=label)
        ax_bottom.fill_between(series.index, series.values, step="mid", alpha=0.18, color=color)

    ax_top.set_yticks([lane_positions[0]] + [lane_positions[step] for step in step_indices])
    ax_top.set_yticklabels(lane_labels)
    ax_top.set_ylabel("Run Activity")
    ax_top.set_title("Full Run Timeline by Step")
    ax_top.grid(True, axis="x", alpha=0.25)
    ax_top.legend(loc="upper right", fontsize=8, ncol=2)

    def format_duration(duration_s: float) -> str:
        if duration_s >= 600:
            return f"{duration_s / 60:.1f}m"
        if duration_s >= 120:
            return f"{duration_s / 60:.1f}m"
        return f"{duration_s:.0f}s"

    def format_time_window(start: pd.Timestamp, end: pd.Timestamp) -> str:
        return f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}"

    phase_short = {
        "bring_up_and_preflight": "bring-up",
        "observed_run_span": "observed",
        "wait_for_generation_buffer": "wait",
        "wait_for_generation_buffer_active": "wait*",
        "run_training": "train",
        "run_training_active": "train*",
        "step_finalize": "final",
        "step_finalize_active": "final*",
    }
    per_lane_summary: Dict[int, List[str]] = {}
    for row in sorted(phase_rows, key=lambda item: (int(item["step_index"]), item["start"])):
        step_index = int(row["step_index"])
        summary = (
            f"{format_time_window(row['start'], row['end'])} "
            f"{phase_short[str(row['phase'])]} "
            f"{format_duration(float(row['duration_s']))}"
        )
        per_lane_summary.setdefault(step_index, []).append(summary)

    ax_side.set_title("Per-Step Summary", loc="left", fontsize=10)
    ax_side.set_xlim(0, 1)
    ax_side.set_ylim(ax_top.get_ylim())
    ax_side.set_xticks([])
    ax_side.set_yticks([])
    for spine in ax_side.spines.values():
        spine.set_visible(False)
    blended = mtransforms.blended_transform_factory(ax_side.transAxes, ax_side.transData)
    for step_index in [0] + step_indices:
        lane_y = lane_positions[step_index]
        summary = " | ".join(per_lane_summary.get(step_index, []))
        ax_side.text(
            0.0,
            lane_y,
            summary,
            transform=blended,
            va="center",
            ha="left",
            fontsize=8,
            family="monospace",
        )

    ax_bottom.set_ylabel("Errors / min")
    ax_bottom.set_xlabel("Time (PDT)")
    ax_bottom.grid(True, alpha=0.25)
    if any(not series.empty for _, series, _ in event_specs):
        ax_bottom.legend(loc="upper left", fontsize=8)

    if x_starts and x_ends:
        ax_top.set_xlim(min(x_starts), max(x_ends))
    ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
    ax_bottom.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_trainer_memory_plot(
    output_path: Path,
    trainer_frames: Dict[str, pd.DataFrame],
    gpu_total_gib: float,
) -> None:
    gpu_reserved = trainer_frames["gpu_reserved"]
    node_reserved = trainer_frames["node_reserved"]
    node_allocated = trainer_frames["node_allocated"]
    tz = gpu_reserved.index.tz
    node_gpu_counts = gpu_reserved.columns.to_series().groupby(lambda label: label.split(":gpu")[0]).size()
    node_capacity_pct = {
        node: gpu_total_gib * float(count) for node, count in node_gpu_counts.items()
    }
    node_reserved_pct = node_reserved.copy()
    node_allocated_pct = node_allocated.copy()
    for column in node_reserved.columns:
        capacity_gib = node_capacity_pct[column]
        node_reserved_pct[column] = node_reserved[column] / capacity_gib * 100.0
        node_allocated_pct[column] = node_allocated[column] / capacity_gib * 100.0
    gpu_reserved_pct = gpu_reserved / gpu_total_gib * 100.0

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(18, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.6]},
    )

    for column in node_reserved_pct.columns:
        ax_top.plot(node_reserved_pct.index, node_reserved_pct[column], linewidth=2.0, label=f"{column} reserved")
        ax_top.plot(
            node_allocated_pct.index,
            node_allocated_pct[column],
            linewidth=1.2,
            linestyle="--",
            alpha=0.85,
            label=f"{column} allocated",
        )

    heatmap = ax_bottom.imshow(
        gpu_reserved_pct.T.values,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        extent=(
            mdates.date2num(gpu_reserved_pct.index[0]),
            mdates.date2num(gpu_reserved_pct.index[-1]),
            -0.5,
            len(gpu_reserved_pct.columns) - 0.5,
        ),
        cmap="magma",
        vmin=0.0,
        vmax=100.0,
    )
    cbar = fig.colorbar(heatmap, ax=ax_bottom)
    cbar.set_label("Reserved % per physical GPU")

    ax_top.set_ylim(0, 100)
    ax_top.set_ylabel("Node Reserved / Allocated %")
    ax_top.set_title("Trainer GPU Memory % (reconstructed from worker memory_events)")
    ax_top.legend(ncol=2, fontsize=8)
    ax_top.grid(True, axis="y", alpha=0.25)

    ax_bottom.set_yticks(range(len(gpu_reserved_pct.columns)))
    ax_bottom.set_yticklabels(list(gpu_reserved_pct.columns), fontsize=8)
    ax_bottom.set_ylabel("Trainer GPU")
    ax_bottom.xaxis_date()
    ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
    ax_bottom.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    ax_bottom.set_xlabel("Time")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_rollout_memory_plot(output_path: Path, rollout_gpu_gib: pd.DataFrame) -> None:
    tz = rollout_gpu_gib.index.tz
    fig, ax = plt.subplots(figsize=(18, 6))
    for column in rollout_gpu_gib.columns:
        ax.plot(rollout_gpu_gib.index, rollout_gpu_gib[column], linewidth=1.5, label=column)
    ax.set_title("Rollout GPU Memory (nvidia-smi sampled)")
    ax.set_ylabel("Used GiB")
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    ax.legend(ncol=4, fontsize=8, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_rollout_kv_usage_plot(output_path: Path, rollout_kv_usage: pd.DataFrame) -> None:
    tz = rollout_kv_usage.index.tz
    fig, ax = plt.subplots(figsize=(18, 6))
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    for idx, column in enumerate(rollout_kv_usage.columns):
        ax.plot(
            rollout_kv_usage.index,
            rollout_kv_usage[column],
            linewidth=2.0,
            label=column,
            color=colors[idx % len(colors)],
        )
    if len(rollout_kv_usage.columns) > 1:
        ax.plot(
            rollout_kv_usage.index,
            rollout_kv_usage.mean(axis=1),
            linewidth=2.2,
            linestyle="--",
            color="#222222",
            label="mean",
        )
    ax.set_title("Rollout KV Cache Usage (%)")
    ax.set_ylabel("KV Usage %")
    ax.set_xlabel("Time (PDT)")
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    ax.legend(loc="upper left", fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_rollout_prefix_cache_hit_rate_plot(output_path: Path, prefix_cache_hit_rate: pd.DataFrame) -> None:
    tz = prefix_cache_hit_rate.index.tz
    fig, ax = plt.subplots(figsize=(18, 6))
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    for idx, column in enumerate(prefix_cache_hit_rate.columns):
        ax.plot(
            prefix_cache_hit_rate.index,
            prefix_cache_hit_rate[column],
            linewidth=2.0,
            label=column,
            color=colors[idx % len(colors)],
        )
    if len(prefix_cache_hit_rate.columns) > 1:
        ax.plot(
            prefix_cache_hit_rate.index,
            prefix_cache_hit_rate.mean(axis=1),
            linewidth=2.2,
            linestyle="--",
            color="#222222",
            label="mean",
        )
    ax.set_title("Rollout Prefix Cache Hit Rate (%)")
    ax.set_ylabel("Hit Rate %")
    ax.set_xlabel("Time (PDT)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    ax.legend(loc="upper left", fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_rollout_num_requests_running_plot(output_path: Path, num_requests_running: pd.DataFrame) -> None:
    tz = num_requests_running.index.tz
    fig, ax = plt.subplots(figsize=(18, 6))
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    for idx, column in enumerate(num_requests_running.columns):
        ax.plot(
            num_requests_running.index,
            num_requests_running[column],
            linewidth=2.0,
            label=column,
            color=colors[idx % len(colors)],
        )
    if len(num_requests_running.columns) > 1:
        ax.plot(
            num_requests_running.index,
            num_requests_running.sum(axis=1),
            linewidth=2.2,
            linestyle="--",
            color="#222222",
            label="sum",
        )
    ax.axhline(256, color="#999999", linewidth=1.2, linestyle=":", label="Harbor max_concurrency=256")
    ax.set_title("Rollout num_requests_running")
    ax.set_ylabel("Running Requests")
    ax.set_xlabel("Time (PDT)")
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    ax.legend(loc="upper left", fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_rollout_num_requests_waiting_plot(output_path: Path, num_requests_waiting: pd.DataFrame) -> None:
    tz = num_requests_waiting.index.tz
    fig, ax = plt.subplots(figsize=(18, 6))
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    for idx, column in enumerate(num_requests_waiting.columns):
        ax.plot(
            num_requests_waiting.index,
            num_requests_waiting[column],
            linewidth=2.0,
            label=column,
            color=colors[idx % len(colors)],
        )
    if len(num_requests_waiting.columns) > 1:
        ax.plot(
            num_requests_waiting.index,
            num_requests_waiting.sum(axis=1),
            linewidth=2.2,
            linestyle="--",
            color="#222222",
            label="sum",
        )
    ax.set_title("Rollout num_requests_waiting")
    ax.set_ylabel("Waiting Requests")
    ax.set_xlabel("Time (PDT)")
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    ax.legend(loc="upper left", fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(
    output_path: Path,
    run_name: str,
    launcher_info: Dict[str, object],
    phase_rows: List[Dict[str, object]],
    trainer_frames: Dict[str, pd.DataFrame],
    rollout_gpu_gib: pd.DataFrame,
    rollout_kv_usage: pd.DataFrame,
    rollout_prefix_cache_hit_rate: pd.DataFrame,
    rollout_num_requests_running: pd.DataFrame,
    rollout_num_requests_waiting: pd.DataFrame,
    thunder_wait_timeout_times: List[pd.Timestamp],
) -> None:
    steps: List[StepInfo] = launcher_info["steps"]  # type: ignore[assignment]
    agent_timeout_times: List[pd.Timestamp] = launcher_info["agent_timeout_times"]  # type: ignore[assignment]
    none_error_times: List[pd.Timestamp] = launcher_info["none_error_times"]  # type: ignore[assignment]
    cancel_time: Optional[pd.Timestamp] = launcher_info["cancel_time"]  # type: ignore[assignment]

    trainer_peak_gpu = trainer_frames["gpu_reserved"].max().sort_values(ascending=False)
    trainer_peak_node = trainer_frames["node_reserved"].max().sort_values(ascending=False)
    rollout_peak_gpu = rollout_gpu_gib.max().sort_values(ascending=False)
    rollout_peak_kv = rollout_kv_usage.max().sort_values(ascending=False)
    rollout_peak_prefix = rollout_prefix_cache_hit_rate.max().sort_values(ascending=False)
    rollout_peak_running = rollout_num_requests_running.max().sort_values(ascending=False)
    rollout_peak_waiting = rollout_num_requests_waiting.max().sort_values(ascending=False)

    with output_path.open("w") as handle:
        handle.write(f"# Full Run Timeline Analysis: {run_name}\n\n")
        handle.write("## Window\n\n")
        handle.write(f"- launcher_start: {launcher_info['first_ts']}\n")
        handle.write(f"- launcher_end: {launcher_info['last_ts']}\n")
        handle.write(f"- cancel_time: {cancel_time}\n")
        handle.write(f"- agent_timeout_count: {len(agent_timeout_times)}\n")
        handle.write(f"- thunder_wait_timeout_count: {len(thunder_wait_timeout_times)}\n")
        handle.write(f"- none_type_trajectory_id_count: {len(none_error_times)}\n\n")

        handle.write("## Phase Table\n\n")
        handle.write("| phase | step | start | end | duration_s | notes |\n")
        handle.write("| --- | --- | --- | --- | ---: | --- |\n")
        for row in phase_rows:
            handle.write(
                f"| {row['phase']} | {row['step_index']} | {row['start']} | {row['end']} | "
                f"{row['duration_s']:.2f} | {row['notes']} |\n"
            )
        handle.write("\n")

        handle.write("## Step Summary\n\n")
        handle.write("| step | wait_s | train_s | finalize_s | global_step_after |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: |\n")
        for step in steps:
            wait_s = (
                (step.wait_end - step.wait_start).total_seconds()
                if step.wait_start is not None and step.wait_end is not None
                else float("nan")
            )
            train_s = (
                (step.train_end - step.train_start).total_seconds()
                if step.train_start is not None and step.train_end is not None
                else float("nan")
            )
            finalize_s = (
                (step.step_end - (step.train_end or step.wait_end)).total_seconds()
                if step.step_end is not None and (step.train_end or step.wait_end) is not None
                else float("nan")
            )
            handle.write(
                f"| {step.step_index} | {wait_s:.2f} | {train_s:.2f} | {finalize_s:.2f} | "
                f"{step.global_step_after if step.global_step_after is not None else ''} |\n"
            )
        handle.write("\n")

        handle.write("## Trainer Peaks\n\n")
        handle.write("| trainer_gpu | peak_reserved_gib |\n")
        handle.write("| --- | ---: |\n")
        for label, value in trainer_peak_gpu.items():
            handle.write(f"| {label} | {value:.2f} |\n")
        handle.write("\n")

        handle.write("| trainer_node | peak_reserved_gib |\n")
        handle.write("| --- | ---: |\n")
        for label, value in trainer_peak_node.items():
            handle.write(f"| {label} | {value:.2f} |\n")
        handle.write("\n")

        handle.write("## Rollout Peaks\n\n")
        handle.write("| rollout_gpu | peak_used_gib |\n")
        handle.write("| --- | ---: |\n")
        for label, value in rollout_peak_gpu.items():
            handle.write(f"| {label} | {value:.2f} |\n")
        handle.write("\n")
        handle.write("| rollout_backend | peak_kv_usage_pct |\n")
        handle.write("| --- | ---: |\n")
        for label, value in rollout_peak_kv.items():
            handle.write(f"| {label} | {value:.2f} |\n")
        handle.write("\n")
        handle.write("| rollout_backend | peak_prefix_cache_hit_rate_pct |\n")
        handle.write("| --- | ---: |\n")
        for label, value in rollout_peak_prefix.items():
            handle.write(f"| {label} | {value:.2f} |\n")
        handle.write("\n")
        handle.write("| rollout_backend | peak_num_requests_running |\n")
        handle.write("| --- | ---: |\n")
        for label, value in rollout_peak_running.items():
            handle.write(f"| {label} | {value:.0f} |\n")
        handle.write("\n")
        handle.write("| rollout_backend | peak_num_requests_waiting |\n")
        handle.write("| --- | ---: |\n")
        for label, value in rollout_peak_waiting.items():
            handle.write(f"| {label} | {value:.0f} |\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a Qwen3 Harbor full-run timeline and GPU memory traces.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--log-root", default="/home/hkang/zthunder_agent/tmp_logs")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--timezone", default="America/Los_Angeles")
    return parser.parse_args()


def resolve_launcher_path(run_dir: Path) -> Path:
    candidates = (
        run_dir / "launcher_train_driver.log",
        run_dir / "launcher-interactive.log",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def main() -> None:
    args = parse_args()
    tz = ZoneInfo(args.timezone)
    run_dir = Path(args.log_root) / args.run_name
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    launcher_path = resolve_launcher_path(run_dir)
    thunder_path = run_dir / "thunderagent.log"
    memory_events_dir = run_dir / "memory_events"
    rollout_gpu_summary_path = run_dir / "rollout" / "monitoring" / "gpu_summary.tsv"
    rollout_vllm_metrics_path = run_dir / "rollout" / "monitoring" / "vllm_metrics.tsv"

    launcher_info = parse_launcher(launcher_path, tz)
    first_ts = launcher_info["first_ts"]
    last_ts = launcher_info["last_ts"]
    inferred_start, inferred_end = infer_monitor_window(run_dir, tz)
    if first_ts is None or last_ts is None:
        first_ts = first_ts or inferred_start
        last_ts = last_ts or inferred_end
    elif not launcher_info["steps"] and inferred_start is not None and inferred_end is not None:
        if (last_ts - first_ts).total_seconds() < 60:
            first_ts = min(first_ts, inferred_start)
            last_ts = max(last_ts, inferred_end)
    if first_ts is None or last_ts is None:
        raise RuntimeError("Failed to parse launcher timestamps or monitoring fallback window")

    thunder_wait_timeout_times = parse_thunder_wait_timeout_times(thunder_path, tz)
    candidate_end_times = [last_ts]
    if thunder_wait_timeout_times:
        candidate_end_times.append(max(thunder_wait_timeout_times))
    cancel_time = launcher_info["cancel_time"]
    if cancel_time is not None:
        candidate_end_times.append(cancel_time)
    end_time = max(candidate_end_times)

    trainer_frames = parse_trainer_memory(memory_events_dir, first_ts, end_time)
    rollout_gpu_gib = parse_rollout_gpu_summary(rollout_gpu_summary_path, first_ts, end_time)
    gpu_total_gib = parse_gpu_total_gib_from_summary(rollout_gpu_summary_path)
    rollout_kv_usage = parse_rollout_kv_usage(rollout_vllm_metrics_path, first_ts, end_time)
    rollout_prefix_cache_hit_rate = parse_rollout_prefix_cache_hit_rate(rollout_vllm_metrics_path, first_ts, end_time)
    rollout_num_requests_running = parse_rollout_num_requests_running(rollout_vllm_metrics_path, first_ts, end_time)
    rollout_num_requests_waiting = parse_rollout_num_requests_waiting(rollout_vllm_metrics_path, first_ts, end_time)
    phase_rows = build_phase_rows(launcher_info["steps"], first_ts, end_time)  # type: ignore[arg-type]
    if not phase_rows:
        phase_rows = [
            {
                "phase": "observed_run_span",
                "step_index": 0,
                "start": first_ts,
                "end": end_time,
                "duration_s": (end_time - first_ts).total_seconds(),
                "notes": "inferred from monitoring timestamps",
            }
        ]

    make_timeline_plot(
        output_dir / "timeline_overview.png",
        phase_rows,
        launcher_info["agent_timeout_times"],  # type: ignore[arg-type]
        thunder_wait_timeout_times,
        launcher_info["none_error_times"],  # type: ignore[arg-type]
        cancel_time,
    )
    make_trainer_memory_plot(output_dir / "trainer_memory.png", trainer_frames, gpu_total_gib)
    make_rollout_memory_plot(output_dir / "rollout_memory.png", rollout_gpu_gib)
    make_rollout_kv_usage_plot(output_dir / "rollout_kv_usage.png", rollout_kv_usage)
    make_rollout_prefix_cache_hit_rate_plot(
        output_dir / "rollout_prefix_cache_hit_rate.png",
        rollout_prefix_cache_hit_rate,
    )
    make_rollout_num_requests_running_plot(
        output_dir / "rollout_num_requests_running.png",
        rollout_num_requests_running,
    )
    make_rollout_num_requests_waiting_plot(
        output_dir / "rollout_num_requests_waiting.png",
        rollout_num_requests_waiting,
    )
    write_report(
        output_dir / "timeline_report.md",
        args.run_name,
        launcher_info,
        phase_rows,
        trainer_frames,
        rollout_gpu_gib,
        rollout_kv_usage,
        rollout_prefix_cache_hit_rate,
        rollout_num_requests_running,
        rollout_num_requests_waiting,
        thunder_wait_timeout_times,
    )

    print(f"analysis_dir={output_dir}")
    print(f"timeline_plot={output_dir / 'timeline_overview.png'}")
    print(f"trainer_plot={output_dir / 'trainer_memory.png'}")
    print(f"rollout_plot={output_dir / 'rollout_memory.png'}")
    print(f"rollout_kv_plot={output_dir / 'rollout_kv_usage.png'}")
    print(f"rollout_prefix_cache_hit_rate_plot={output_dir / 'rollout_prefix_cache_hit_rate.png'}")
    print(f"rollout_num_requests_running_plot={output_dir / 'rollout_num_requests_running.png'}")
    print(f"rollout_num_requests_waiting_plot={output_dir / 'rollout_num_requests_waiting.png'}")
    print(f"report={output_dir / 'timeline_report.md'}")


if __name__ == "__main__":
    main()
