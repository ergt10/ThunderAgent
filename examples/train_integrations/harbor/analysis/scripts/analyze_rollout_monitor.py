#!/usr/bin/env python3
"""Analyze rollout-node monitor outputs and generate a readable summary figure."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


TIME_FMT = "%Y-%m-%dT%H:%M:%S%z"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path, help="Run dir or rollout log dir")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def resolve_dirs(run_dir: Path) -> tuple[Path, Path]:
    monitoring_dir = run_dir / "monitoring"
    if monitoring_dir.exists():
        output_dir = run_dir / "analysis"
        return monitoring_dir, output_dir

    rollout_dir = run_dir / "rollout"
    monitoring_dir = rollout_dir / "monitoring"
    if monitoring_dir.exists():
        output_dir = rollout_dir / "analysis"
        return monitoring_dir, output_dir

    raise FileNotFoundError(f"Could not find monitoring dir under {run_dir}")


def parse_timestamp(value: str) -> float:
    return datetime.strptime(value, TIME_FMT).timestamp()


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_summary(
    path: Path,
    figure_path: Path,
    gpu_peaks: list[tuple[str, float]],
    process_peaks: list[tuple[str, float, str, str]],
    kv_peaks: list[tuple[str, float]],
) -> None:
    lines = [
        "# Rollout Monitor Summary",
        "",
        f"Figure: `{figure_path}`",
        "",
        "## GPU Peaks",
        "",
    ]
    for gpu_index, peak_mib in gpu_peaks:
        lines.append(f"- GPU {gpu_index}: `{peak_mib:.0f} MiB` peak used")

    lines.extend(["", "## Process Peaks", ""])
    for pid, peak_mib, gpu_index, role in process_peaks:
        lines.append(f"- PID `{pid}` on GPU `{gpu_index}` (`{role}`): `{peak_mib:.0f} MiB` peak used")

    lines.extend(["", "## KV Cache Peaks", ""])
    if kv_peaks:
        for source, peak_pct in kv_peaks:
            lines.append(f"- `{source}`: `{peak_pct:.2f}%` peak KV cache usage")
    else:
        lines.append("- No KV cache metrics captured")

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    monitoring_dir, default_output_dir = resolve_dirs(run_dir)
    output_dir = args.output_dir.resolve() if args.output_dir else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_rows = read_tsv(monitoring_dir / "gpu_summary.tsv")
    proc_rows = read_tsv(monitoring_dir / "gpu_processes.tsv")
    kv_rows = read_tsv(monitoring_dir / "vllm_metrics.tsv")

    fig, axes = plt.subplots(3, 1, figsize=(15, 11), constrained_layout=True)

    gpu_series: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    gpu_peaks: list[tuple[str, float]] = []
    for row in gpu_rows:
        gpu_index = row["gpu_index"]
        t, y = gpu_series[gpu_index]
        t.append(parse_timestamp(row["timestamp"]))
        y.append(float(row["memory_used_mib"]))
    for gpu_index in sorted(gpu_series, key=lambda x: int(x)):
        t, y = gpu_series[gpu_index]
        if not t:
            continue
        t0 = t[0]
        rel_t = [(cur - t0) / 60.0 for cur in t]
        axes[0].plot(rel_t, y, label=f"GPU {gpu_index}", linewidth=1.8)
        gpu_peaks.append((gpu_index, max(y)))
    axes[0].set_title("Rollout Node Per-GPU Memory Timeline")
    axes[0].set_xlabel("Minutes")
    axes[0].set_ylabel("GPU memory used (MiB)")
    axes[0].legend(ncol=4, fontsize=8)
    axes[0].grid(alpha=0.2)

    proc_series: dict[str, dict[str, object]] = {}
    for row in proc_rows:
        pid = row["pid"]
        entry = proc_series.setdefault(
            pid,
            {
                "times": [],
                "values": [],
                "gpu_index": row.get("gpu_index", ""),
                "role": row.get("role", ""),
                "process_name": row.get("process_name", ""),
            },
        )
        entry["times"].append(parse_timestamp(row["timestamp"]))
        entry["values"].append(float(row["used_gpu_memory_mib"]))
    process_peaks = sorted(
        (
            (
                pid,
                max(entry["values"]),  # type: ignore[arg-type]
                str(entry["gpu_index"]),
                str(entry["role"]),
                str(entry["process_name"]),
            )
            for pid, entry in proc_series.items()
            if entry["times"]
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    top_processes = process_peaks[:8]
    for pid, _peak_mib, gpu_index, role, process_name in top_processes:
        entry = proc_series[pid]
        times = entry["times"]  # type: ignore[assignment]
        values = entry["values"]  # type: ignore[assignment]
        t0 = times[0]
        rel_t = [(cur - t0) / 60.0 for cur in times]
        axes[1].plot(rel_t, values, linewidth=1.7, label=f"{process_name}:{pid} gpu{gpu_index} {role}")
    axes[1].set_title("Rollout Process GPU Memory Timeline (Top 8 Peaks)")
    axes[1].set_xlabel("Minutes")
    axes[1].set_ylabel("Process GPU memory (MiB)")
    if top_processes:
        axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.2)

    kv_series: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    kv_peaks: list[tuple[str, float]] = []
    for row in kv_rows:
        if not row.get("kv_cache_usage_pct"):
            continue
        source = row["source"]
        t, y = kv_series[source]
        t.append(parse_timestamp(row["timestamp"]))
        y.append(float(row["kv_cache_usage_pct"]))
    for source in sorted(kv_series):
        t, y = kv_series[source]
        if not t:
            continue
        t0 = t[0]
        rel_t = [(cur - t0) / 60.0 for cur in t]
        axes[2].plot(rel_t, y, linewidth=1.8, label=source)
        kv_peaks.append((source, max(y)))
    axes[2].set_title("vLLM KV Cache Usage")
    axes[2].set_xlabel("Minutes")
    axes[2].set_ylabel("KV cache usage (%)")
    if kv_series:
        axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.2)

    figure_path = output_dir / "rollout_monitor_overview.png"
    fig.savefig(figure_path, dpi=args.dpi)
    plt.close(fig)

    write_summary(
        output_dir / "rollout_monitor_summary.md",
        figure_path,
        sorted(gpu_peaks, key=lambda item: int(item[0])),
        [(pid, peak, gpu_index, role) for pid, peak, gpu_index, role, _name in top_processes],
        kv_peaks,
    )
    print(f"wrote {figure_path}")
    print(f"wrote {output_dir / 'rollout_monitor_summary.md'}")


if __name__ == "__main__":
    main()
