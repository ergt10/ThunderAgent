#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlparse

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ThunderAgent and trial-progress monitoring outputs.")
    parser.add_argument("--log-dir", required=True, help="Run log directory containing monitoring/*.tsv")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for plots/reports. Defaults to <log-dir>/analysis",
    )
    return parser.parse_args()


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path, sep="\t")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%dT%H:%M:%S%z", errors="coerce")
        df = df.dropna(subset=["timestamp"])
    if "event_timestamp" in df.columns:
        df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], format="%Y-%m-%dT%H:%M:%S%z", errors="coerce")
    return df


def backend_label(backend_url: str) -> str:
    parsed = urlparse(str(backend_url))
    port = parsed.port
    if port is None:
        return str(backend_url)
    return f"{parsed.hostname}:{port}"


def setup_time_axis(ax) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.35)


def safe_int(value) -> int:
    if pd.isna(value):
        return 0
    try:
        return int(value)
    except Exception:
        return 0


def plot_backend_tokens(backend_df: pd.DataFrame, events_df: pd.DataFrame, output_path: Path) -> None:
    if backend_df.empty:
        return

    backends = list(dict.fromkeys(backend_df["backend_url"].tolist()))
    fig, axes = plt.subplots(len(backends), 1, figsize=(16, 4.2 * len(backends)), sharex=True, constrained_layout=True)
    if len(backends) == 1:
        axes = [axes]

    for ax, backend_url in zip(axes, backends):
        cur = backend_df[backend_df["backend_url"] == backend_url].sort_values("timestamp")
        label = backend_label(backend_url)
        ax.plot(cur["timestamp"], cur["reasoning_program_tokens"], label="reasoning tokens", color="#D1495B", linewidth=2.0)
        ax.plot(cur["timestamp"], cur["acting_program_tokens"], label="acting tokens", color="#3B82F6", linewidth=2.0)
        ax.plot(cur["timestamp"], cur["active_program_tokens"], label="active tokens", color="#2F855A", linewidth=1.5, alpha=0.85)

        y_max = max(
            1.0,
            float(cur["active_program_tokens"].max()),
            float(cur["reasoning_program_tokens"].max()),
            float(cur["acting_program_tokens"].max()),
        )
        backend_events = events_df[events_df["backend_url"] == backend_url].copy()
        if not backend_events.empty:
            pauses = backend_events[backend_events["event_type"] == "paused"]
            resumes = backend_events[backend_events["event_type"] == "resumed"]
            if not pauses.empty:
                ax.vlines(pauses["event_timestamp"], 0, y_max, color="#E11D48", alpha=0.18, linewidth=1.0, label="pause")
            if not resumes.empty:
                ax.vlines(resumes["event_timestamp"], 0, y_max, color="#16A34A", alpha=0.18, linewidth=1.0, label="resume")

        ax.set_title(f"ThunderAgent backend tokens: {label}")
        ax.set_ylabel("tokens")
        setup_time_axis(ax)
        ax.legend(loc="upper left", ncol=5, frameon=False)

    axes[-1].set_xlabel("time")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_program_counts(program_df: pd.DataFrame, output_path: Path) -> None:
    if program_df.empty:
        return

    df = program_df.sort_values("timestamp").copy()
    fig, ax = plt.subplots(figsize=(16, 7), constrained_layout=True)
    ax.plot(df["timestamp"], df["reasoning_programs"], label="reasoning programs", color="#D1495B", linewidth=2.0)
    ax.plot(df["timestamp"], df["acting_programs"], label="acting programs", color="#3B82F6", linewidth=2.0)
    ax.plot(df["timestamp"], df["paused_programs"], label="paused programs", color="#A855F7", linewidth=1.8)
    ax.plot(df["timestamp"], df["total_programs"], label="total tracked programs", color="#2F855A", linewidth=1.5, alpha=0.85)
    ax.set_ylabel("current program count")
    setup_time_axis(ax)

    ax2 = ax.twinx()
    ax2.plot(df["timestamp"], df["released_total"], label="released total", color="#111827", linewidth=2.0, linestyle="--")
    ax2.set_ylabel("cumulative released programs")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left", ncol=3, frameon=False)
    ax.set_title("ThunderAgent global program state over time")
    ax.set_xlabel("time")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_trial_progress(progress_df: pd.DataFrame, output_path: Path) -> None:
    if progress_df.empty:
        return

    df = progress_df.sort_values("timestamp").copy()
    fig, ax = plt.subplots(figsize=(16, 6), constrained_layout=True)
    ax.plot(df["timestamp"], df["completed_trials_count"], label="completed trials", color="#111827", linewidth=2.2)
    ax.plot(df["timestamp"], df["result_json_count"], label="result.json count", color="#2563EB", linewidth=1.8)
    ax.plot(df["timestamp"], df["exception_txt_count"], label="exception.txt count", color="#DC2626", linewidth=1.6)
    ax.plot(df["timestamp"], df["trajectory_json_count"], label="trajectory.json count", color="#16A34A", linewidth=1.4)
    ax.set_title("SkyRL completed-trial progress over time")
    ax.set_ylabel("count")
    ax.set_xlabel("time")
    setup_time_axis(ax)
    ax.legend(loc="upper left", ncol=4, frameon=False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(
    backend_df: pd.DataFrame,
    program_df: pd.DataFrame,
    events_df: pd.DataFrame,
    progress_df: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = ["# ThunderAgent Monitoring Report", ""]

    if not backend_df.empty:
        lines.extend(["## Backend Peaks", ""])
        for backend_url in dict.fromkeys(backend_df["backend_url"].tolist()):
            cur = backend_df[backend_df["backend_url"] == backend_url]
            lines.append(
                "- {}: reasoning_tokens_peak={}, acting_tokens_peak={}, active_tokens_peak={}, reasoning_count_peak={}, acting_count_peak={}, paused_count_peak={}".format(
                    backend_label(backend_url),
                    safe_int(cur["reasoning_program_tokens"].max()),
                    safe_int(cur["acting_program_tokens"].max()),
                    safe_int(cur["active_program_tokens"].max()),
                    safe_int(cur["reasoning_program_count"].max()),
                    safe_int(cur["acting_program_count"].max()),
                    safe_int(cur["paused_program_count"].max()),
                )
            )
        lines.append("")

    if not program_df.empty:
        last = program_df.sort_values("timestamp").iloc[-1]
        lines.extend(
            [
                "## Final Global Program State",
                "",
                f"- total_programs={safe_int(last['total_programs'])}",
                f"- reasoning_programs={safe_int(last['reasoning_programs'])}",
                f"- acting_programs={safe_int(last['acting_programs'])}",
                f"- paused_programs={safe_int(last['paused_programs'])}",
                f"- released_total={safe_int(last['released_total'])}",
                "",
            ]
        )

    if not events_df.empty:
        counts = events_df["event_type"].value_counts().sort_index()
        lines.extend(["## Event Counts", ""])
        for event_type, count in counts.items():
            lines.append(f"- {event_type}={int(count)}")
        lines.append("")

    if not progress_df.empty:
        last = progress_df.sort_values("timestamp").iloc[-1]
        lines.extend(
            [
                "## Final Trial Progress",
                "",
                f"- trial_dirs={safe_int(last['trial_dirs'])}",
                f"- completed_trials_count={safe_int(last['completed_trials_count'])}",
                f"- result_json_count={safe_int(last['result_json_count'])}",
                f"- exception_txt_count={safe_int(last['exception_txt_count'])}",
                f"- trajectory_json_count={safe_int(last['trajectory_json_count'])}",
                "",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    log_dir = Path(args.log_dir).resolve()
    monitoring_dir = log_dir / "monitoring"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else log_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    backend_df = read_tsv(monitoring_dir / "thunderagent_backend_state.tsv")
    program_df = read_tsv(monitoring_dir / "thunderagent_program_state.tsv")
    events_df = read_tsv(monitoring_dir / "thunderagent_events.tsv")
    progress_df = read_tsv(monitoring_dir / "trial_progress.tsv")

    if not events_df.empty and "event_timestamp" in events_df.columns:
        events_df = events_df.dropna(subset=["event_timestamp"]).sort_values("event_timestamp")

    plot_backend_tokens(backend_df, events_df, output_dir / "thunderagent_backend_tokens.png")
    plot_program_counts(program_df, output_dir / "thunderagent_program_counts.png")
    plot_trial_progress(progress_df, output_dir / "skyrl_completed_trials.png")
    write_report(
        backend_df,
        program_df,
        events_df,
        progress_df,
        output_dir / "thunderagent_monitoring_report.md",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
